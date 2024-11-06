def test_import() -> None:
    """Test if the app can be imported.

    Expected behavior: No exceptions are raised (mainly circular import errors).
    """

    import app

    assert app
