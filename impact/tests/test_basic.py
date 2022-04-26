from impact import Impact


def test_basic_init_and_run(lcls_archive_file):
    I = Impact(verbose=True)
    I.load_archive(lcls_archive_file)

    # Switches for MPI
    I.numprocs = 1
    I.total_charge = 0
    I.run()
