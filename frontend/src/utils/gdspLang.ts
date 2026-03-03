import { StreamLanguage, type StreamParser } from "@codemirror/language";

// Statement-starting keywords
const KEYWORDS = new Set([
  "graph", "param", "in", "out", "control", "import", "const",
  "feedback", "buffer",
]);

// Named constants
const NAMED_CONSTANTS = new Set([
  "pi", "twopi", "halfpi", "invpi", "e", "phi",
  "ln2", "ln10", "log2e", "log10e", "sqrt2", "sqrt1_2",
  "t60", "t60time", "true", "false",
]);

// Node-type builtins usable as function calls.
// In oneDark: "builtin" -> standard(variableName) -> whiskey/orange (#d19a66)
const BUILTINS = new Set([
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
]);

// oneDark color mapping for reference:
//   keyword     -> violet   (#c678dd)  -- graph, param, in, out, buffer ...
//   builtin     -> whiskey  (#d19a66)  -- phasor, buf_read, svf ...
//   atom        -> whiskey  (#d19a66)  -- pi, twopi, true, false ...
//   number      -> chalky   (#e5c07b)  -- 440, 0.5, 1e3
//   comment     -> stone    (#7d8799)  -- # comments
//   string      -> sage     (#98c379)  -- "strings"
//   variableName -> coral   (#e06c75)  -- user identifiers
//   meta        -> stone    (#7d8799)  -- @control
//   punctuation -> ivory                -- { } ( ) , ; ..

const gdspParser: StreamParser<null> = {
  startState: () => null,

  token(stream) {
    // Skip whitespace
    if (stream.eatSpace()) return null;

    // Line comments: # to end of line
    if (stream.match("#")) {
      stream.skipToEnd();
      return "comment";
    }

    // Strings
    if (stream.peek() === '"') {
      stream.next();
      while (!stream.eol()) {
        const ch = stream.next();
        if (ch === '"') break;
        if (ch === "\\") stream.next();
      }
      return "string";
    }

    // Range operator .. (check BEFORE numbers so 0.1..20000 tokenizes correctly)
    if (stream.match("..")) return "punctuation";

    // Numbers: float or integer. Don't match a leading minus (that's a unary op).
    // Match float first (greedy: 0.1 before 0), but stop before .. range.
    if (stream.match(/^\d+\.\d+([eE][+-]?\d+)?/) || stream.match(/^\d+([eE][+-]?\d+)?/)) {
      return "number";
    }

    // @control prefix
    if (stream.match("@")) return "meta";

    // Operators
    if (stream.match(/^[+\-*\/%=<>!&|^~?:]/)) return "operator";

    // Punctuation
    if (stream.match(/^[{}(),;]/)) return "punctuation";

    // Identifiers, keywords, builtins
    if (stream.match(/^[a-zA-Z_][a-zA-Z0-9_]*/)) {
      const word = stream.current();
      if (KEYWORDS.has(word)) return "keyword";
      if (NAMED_CONSTANTS.has(word)) return "atom";
      if (BUILTINS.has(word)) return "builtin";
      return "variableName";
    }

    // Skip unknown chars
    stream.next();
    return null;
  },
};

export const gdspLanguage = StreamLanguage.define(gdspParser);
