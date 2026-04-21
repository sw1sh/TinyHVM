(* Graphviz reference PNGs for side-by-side comparison with *_wl.png.
   Requires `dot` on PATH. From repo root:
     wolframscript -f wl/examples/thvm_graphs/render_gv_png.wl *)

base = Which[
   StringQ[$InputFileName] && $InputFileName =!= "" && $InputFileName =!= $Failed,
    DirectoryName[$InputFileName],
   ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] >= 1,
    DirectoryName[ExpandFileName@First[$ScriptCommandLine]],
   True, Quiet@Check[NotebookDirectory[], $Failed]];
If[! StringQ[base], Print["Run with wolframscript -f render_gv_png.wl"]; Abort[]];

paths = FileNames[FileNameJoin@{base, "*", "*.dot"}];
Do[
  p = paths[[k]];
  out = StringReplace[p, ".dot" -> "_gv.png"];
  r = Run["/opt/homebrew/bin/dot -Tpng -Gdpi=120 \"" <> p <> "\" -o \"" <> out <> "\""];
  If[r =!= 0, Run["dot -Tpng -Gdpi=120 \"" <> p <> "\" -o \"" <> out <> "\""]],
  {k, Length[paths]}];

Print["Wrote ", Length[paths], " *_gv.png next to each .dot under ", base];
