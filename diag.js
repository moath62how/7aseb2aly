// Diagnostic: check activation/gradient health layer by layer.
const fs = require("fs");
const path = require("path");
process.env.DIAG = "1";
const src = fs.readFileSync(path.join(__dirname, "train.js"), "utf8");
// reuse the loader by evaluating just the parse part is messy; simpler: re-implement tiny probe
// We instead import train.js pieces is not possible (it runs training). So do a self-contained probe:
const EP = parseFloat(process.env.EPOCHS || "0");
console.log("diagnostic placeholder, EPOCHS=" + EP);
