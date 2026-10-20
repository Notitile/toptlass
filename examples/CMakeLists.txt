

set(TOPTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

function(toptlass_example_add_executable NAME)



foreach(EXAMPLE 
        00_basic_gemm
        )

        add_subdirectory(${EXAMPLE})
    
endforeach()


