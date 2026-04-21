(* Rasterize TinyHVM *.dot dumps with ImportDOT (Wolfram Graph + Graphviz layout).
   Usage from repo root:
     wolframscript -f wl/examples/thvm_graphs/render_wl_png.wl
   Or in a notebook: Get["wl/examples/thvm_graphs/render_wl_png.wl"] *)

base = Which[
   StringQ[$InputFileName] && $InputFileName =!= "" && $InputFileName =!= $Failed,
    DirectoryName[$InputFileName],
   ListQ[$ScriptCommandLine] && Length[$ScriptCommandLine] >= 1,
    DirectoryName[ExpandFileName@First[$ScriptCommandLine]],
   True, Quiet@Check[NotebookDirectory[], $Failed]];
If[! StringQ[base],
  Print["Set directory: use wolframscript -f render_wl_png.wl or Get from notebook"];
  Abort[]];

kernel = FileNameJoin@{base, "..", "..", "Kernel", "ImportCallGraphDOT.wl"};
If[! FileExistsQ[kernel],
  Print["Cannot find ImportCallGraphDOT.wl at ", kernel]; Abort[]];
Get[kernel];
paths = FileNames[FileNameJoin@{base, "*", "*.dot"}];
Do[
  p = paths[[k]];
  g = Quiet@ImportDOT[p, Method -> "GraphvizGraphics"];
  out = StringReplace[p, ".dot" -> "_wl.png"];
  If[GraphQ[g] || Head[g] === Graphics || Head[g] === GraphicsBox,
   img = Rasterize[
     Show[g, PerformanceGoal -> "Quality"],
     ImageResolution -> 144, Background -> White];
   Export[out, ImageResize[img, {1000, Automatic}]],
   Print["ImportDOT failed: ", p, " -> ", g]],
  {k, Length[paths]}];

Print["Wrote ", Length[paths], " *_wl.png under ", base];
