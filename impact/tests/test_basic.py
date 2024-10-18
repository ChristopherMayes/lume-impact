from impact import Impact
import os


def test_basic_init_run_archive():
    I = Impact()
    # Switches to make this fast
    I.header["Np"] = 100
    I.header["Bcurr"] = 0  # turn off SC
    I.run()

    # Use the tempdir as scratch
    afile = os.path.join(I.path, "test.h5")
    I.archive(afile)
    Impact.from_archive(afile)


def test_plot():
    I = Impact()
    # Switches to make this fast
    I.header["Np"] = 100
    I.header["Bcurr"] = 0  # turn off SC
    I.run()
    I.plot()
