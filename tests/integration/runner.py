import os

from gar1.itest import gar1_itest
from gar2.itest import gar2_itest
from gr.itest import gr_itest
from gs.itest import gs_itest
from gst.itest import gst_itest
from gstcoreg2.itest import gstcoreg2_itest
from gstsmall.itest import gstsmall_itest
from par1.itest import par1_itest
from pr.itest import pr_itest
from pst.itest import pst_itest

# run_test_scripts = {
#     "gst/itest.py": ["seq", "par_f", "par_s"],
#     "gcoreg/itest.py": ["seq", "par_f", "par_s"],
#     "par1/itest.py": ["seq", "par_f"],
#     "pr/itest.py": ["seq"],
# }

itest_calls = {
    gst_itest: ["seq"],
    gstcoreg2_itest: ["seq"],
    par1_itest: ["seq"],
    pr_itest: ["seq"],
    gr_itest: ["seq"],
    gar1_itest: ["seq"],
    gar2_itest: ["seq"],
    gs_itest: ["seq"],
    gstsmall_itest: ["seq"],
    pst_itest: ["seq"],
}

os.environ["ARRAY_MODULE"] = "cupy"  # "numpy" or "cupy"

if __name__ == "__main__":

    for itest, modes in itest_calls.items():
        print(f"{itest.__name__} in mode `{modes[0]}` returned: {itest()}")

        # for mode in modes:
        #     print(f"Running {itest.__name__} in {mode} mode...")
        #     command = f"ARRAY_MODULE={mode} python tests/integration/{script}"
        #     os.system(command)