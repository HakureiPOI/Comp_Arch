add_test([=[gemm_kernel_opt_loop_unrolling.test0]=]  /workspace/Comp_Arch/lab3/build/dist/bins/lab3_gemm_opt_loop_unrolling.unittest [==[--gtest_filter=gemm_kernel_opt_loop_unrolling.test0]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[gemm_kernel_opt_loop_unrolling.test0]=]  PROPERTIES WORKING_DIRECTORY /workspace/Comp_Arch/lab3/build/src/lab3 SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[gemm_kernel_opt_loop_unrolling.test1]=]  /workspace/Comp_Arch/lab3/build/dist/bins/lab3_gemm_opt_loop_unrolling.unittest [==[--gtest_filter=gemm_kernel_opt_loop_unrolling.test1]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[gemm_kernel_opt_loop_unrolling.test1]=]  PROPERTIES WORKING_DIRECTORY /workspace/Comp_Arch/lab3/build/src/lab3 SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test([=[gemm_kernel_opt_loop_unrolling.test2]=]  /workspace/Comp_Arch/lab3/build/dist/bins/lab3_gemm_opt_loop_unrolling.unittest [==[--gtest_filter=gemm_kernel_opt_loop_unrolling.test2]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[gemm_kernel_opt_loop_unrolling.test2]=]  PROPERTIES WORKING_DIRECTORY /workspace/Comp_Arch/lab3/build/src/lab3 SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  lab3_gemm_opt_loop_unrolling.unittest_TESTS gemm_kernel_opt_loop_unrolling.test0 gemm_kernel_opt_loop_unrolling.test1 gemm_kernel_opt_loop_unrolling.test2)