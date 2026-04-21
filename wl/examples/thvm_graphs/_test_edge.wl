Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "..", "Kernel", "ImportCallGraphDOT.wl"}]];
g = ImportDOT[FileNameJoin[{DirectoryName[$InputFileName], "tiny_linear_bias", "thvm_0_pre_reduce.dot"}], Method -> "Wolfram"];
(* sample vertex colors *)
Print["n38 fill ", GraphPlot`Private`getFillColor /.
  DownValues[ImportDOT] (* wrong *);
Print["done"];
