def test_pyclm():

    try:
        import pyclm
        pyclm.run_pyclm("")

    except Exception as e:

        if isinstance(e, ImportError):
            assert False

        if isinstance(e, ModuleNotFoundError):
            assert False

    assert True


if __name__ == "__main__":
    test_pyclm()