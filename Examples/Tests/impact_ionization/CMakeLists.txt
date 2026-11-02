# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_impact_ionization  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_impact_ionization  # inputs
    "analysis_impact_ionization.py"  # analysis
    "analysis_default_regression.py --path diags/diag"  # checksum
    OFF  # dependency
)
