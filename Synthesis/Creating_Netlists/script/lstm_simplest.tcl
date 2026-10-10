# Versão corrigida e SIMPLES:
set USER aluno6
set PROJECT_DIR /prj/ci/workarea/aluno6/Verilog/lstm_implementation_hdl-main/Synthesis/Creating_Netlists
set TECH_DIR /pdk/gpdk045
set HDL_NAME "lstm_network"

set HDL_FILES [list \
    ${PROJECT_DIR}/RTL/Simplest/lstm_network.sv \
    ${PROJECT_DIR}/RTL/Simplest/lstm.sv \
    ${PROJECT_DIR}/RTL/Simplest/linear_layer.sv \
    ${PROJECT_DIR}/RTL/Simplest/relu_layer.sv \
    ${PROJECT_DIR}/RTL/Simplest/sigmoid_layer.sv \
    ${PROJECT_DIR}/RTL/Simplest/tanh_layer.sv \
    ${PROJECT_DIR}/RTL/Simplest/regbank.sv \
]

set LIB_DIR ${TECH_DIR}/gsclib045_svt_v4.7/gsclib045/timing
set BEST_LIST {fast_vdd1v2_basicCells.lib}
set REPORT_FOLDER ${PROJECT_DIR}/reports_simplest_lstm
set LEC_FOLDER ${REPORT_FOLDER}/lec
set LOG_FOLDER ${REPORT_FOLDER}/logs

set_db auto_ungroup none

set_db [get_db hinsts top/* -match_hier] .ungroup_ok false

set_db init_hdl_search_path "${PROJECT_DIR}"
set_db lib_search_path "${LIB_DIR}"
set_db library "${BEST_LIST}"

read_hdl -sv ${HDL_FILES}


elaborate ${HDL_NAME}

check_design -unresolved ${HDL_NAME}

read_sdc ${PROJECT_DIR}/constraints/constraints.sdc

syn_generic ${HDL_NAME}
write_reports -directory ${REPORT_FOLDER}/generic -tag generic

syn_map ${HDL_NAME}
write_reports -directory ${REPORT_FOLDER}/map -tag mapped
write_do_lec -golden_design rtl -revised_design fv_map -logfile ${LOG_FOLDER}/rtl_to_fv_map.log > ${LEC_FOLDER}/rtl_to_fv_map.tcl

syn_opt
write_do_lec -golden_design fv_map -revised_design ${PROJECT_DIR}/Netlists/lstm_simplest.v -logfile ${LOG_FOLDER}/fv_map_to_final.log > ${LEC_FOLDER}/fv_map_to_final.tcl
write_reports -directory ${REPORT_FOLDER}/genus -tag placed

report_timing > reports_simplest_lstm/report_timing.rpt
report_power  > reports_simplest_lstm/report_power.rpt
report_area   > reports_simplest_lstm/report_area.rpt
report_qor    > reports_simplest_lstm/report_qor.rpt

write_netlist > ${PROJECT_DIR}/Netlists/lstm_simplest.v
write_sdc > ${PROJECT_DIR}/Netlists/lstm_simplest.sdc
write_sdf -nonegchecks -edges check_edge -timescale ns -recrem split -setuphold split > ${PROJECT_DIR}/Netlists/lstm_simplest.sdf
write_scandef > ${PROJECT_DIR}/Netlists/lstm_simplest_scanDEF.scandef


