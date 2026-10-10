set PROJECT_DIR /prj/ci/workarea/aluno6/Verilog/lstm_implementation_hdl-main/Synthesis/Creating_Netlists

set_db lib_search_path /pdk/gpdk045/gsclib045_svt_v4.7/gsclib045/timing
set_db library {fast_vdd1v2_basicCells.lib}

read_hdl ${PROJECT_DIR}/Netlists/lstm_simplified.v
elaborate lstm_network

gui_show
