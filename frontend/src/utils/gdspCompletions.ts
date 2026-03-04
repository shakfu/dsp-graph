import {
  type Completion,
  type CompletionContext,
  type CompletionResult,
  type CompletionSource,
  snippet,
} from "@codemirror/autocomplete";

// Re-use the token sets from gdspLang (they're simple Sets, so we inline
// matching arrays here to avoid coupling to the StreamParser module).

const KEYWORDS = [
  "graph", "param", "in", "out", "control", "import", "const",
  "feedback", "buffer",
];

const NAMED_CONSTANTS = [
  "pi", "twopi", "halfpi", "invpi", "e", "phi",
  "ln2", "ln10", "log2e", "log10e", "sqrt2", "sqrt1_2",
  "t60", "t60time", "true", "false",
];

const BUILTINS = [
  // Oscillators
  "phasor", "sinosc", "sawosc", "pulseosc", "triosc", "cycle", "noise",
  // State / memory
  "history", "delay", "delay_read", "delay_write", "latch", "sample_hold",
  "accum", "counter", "change", "delta", "elapsed", "rate_div",
  // Filters
  "biquad", "svf", "onepole", "dcblock", "allpass", "slide", "smooth",
  "smoothstep",
  // Math / arithmetic
  "add", "sub", "mul", "div", "mod", "pow", "min", "max",
  "abs", "neg", "sqrt", "exp", "exp2", "log", "log2", "log10",
  "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
  "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
  "ceil", "floor", "round", "trunc", "fract", "sign",
  "hypot", "absdiff", "step", "clamp", "wrap", "fold",
  "fastsin", "fastcos", "fasttan", "fastexp", "fastpow",
  // Logic / comparison
  "and", "or", "xor", "not", "bool",
  "gt", "gte", "lt", "lte", "eq", "neq",
  "gtp", "gtep", "ltp", "ltep", "eqp", "neqp",
  "select", "selector",
  // Conversion
  "mtof", "ftom", "dbtoa", "atodb", "degrees", "radians",
  "degtorad", "radtodeg", "mstosamps", "sampstoms",
  "phasewrap", "fixnan", "fixdenorm", "isnan", "isdenorm",
  // Utility
  "scale", "mix", "pass", "mulaccum", "splat",
  "lookup", "wave", "peek", "buf_read", "buf_write", "buf_size",
  "gate_route", "gate_out", "constant", "samplerate",
  // ADSR
  "adsr",
  // Subgraph
  "subgraph",
  // Reverse ops
  "rsub", "rdiv", "rmod",
];

const SNIPPETS = [
  {
    label: "graph",
    detail: "graph block",
    type: "keyword" as const,
    apply: snippet("graph ${name} {\n  ${}\n}"),
  },
  {
    label: "param",
    detail: "parameter declaration",
    type: "keyword" as const,
    apply: snippet("param ${name} ${min}..${max} = ${default}"),
  },
  {
    label: "in",
    detail: "input declaration",
    type: "keyword" as const,
    apply: snippet("in ${name}"),
  },
  {
    label: "out",
    detail: "output declaration",
    type: "keyword" as const,
    apply: snippet("out ${name}"),
  },
  {
    label: "feedback",
    detail: "feedback declaration",
    type: "keyword" as const,
    apply: snippet("feedback ${name}"),
  },
];

/**
 * Creates a gdsp CompletionSource. Pass a function that returns current
 * node IDs from the graph store for dynamic completions.
 */
export function gdspCompletionSource(
  getNodeIds: () => string[],
): CompletionSource {
  return (context: CompletionContext): CompletionResult | null => {
    const word = context.matchBefore(/[a-zA-Z_]\w*/);
    if (!word && !context.explicit) return null;

    const from = word ? word.from : context.pos;
    const prefix = word ? word.text.toLowerCase() : "";

    const options: Completion[] = [];

    // Snippets (only at start of word, prioritized via boost)
    for (const s of SNIPPETS) {
      if (!prefix || s.label.startsWith(prefix)) {
        options.push({ ...s, boost: 10 });
      }
    }

    // Keywords
    for (const kw of KEYWORDS) {
      // Skip keywords that have snippet equivalents
      if (SNIPPETS.some((s) => s.label === kw)) continue;
      if (!prefix || kw.startsWith(prefix)) {
        options.push({ label: kw, type: "keyword", boost: 5 });
      }
    }

    // Builtins
    for (const b of BUILTINS) {
      if (!prefix || b.startsWith(prefix)) {
        options.push({ label: b, type: "function", boost: 3 });
      }
    }

    // Named constants
    for (const c of NAMED_CONSTANTS) {
      if (!prefix || c.startsWith(prefix)) {
        options.push({ label: c, type: "constant", boost: 2 });
      }
    }

    // Dynamic node IDs
    for (const id of getNodeIds()) {
      if (!prefix || id.toLowerCase().startsWith(prefix)) {
        options.push({ label: id, type: "variable", boost: 1 });
      }
    }

    if (options.length === 0) return null;

    return { from, options, validFor: /^\w*$/ };
  };
}
