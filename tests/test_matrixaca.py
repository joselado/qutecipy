import numpy as np

from qutecipy.matrix.aca import MatrixACA


def test_3x3_real():
    A = np.array([
        [1.0, 0.1, -1.0],
        [-0.1, 2.0, -1.0],
        [0.5, 0.2, 0.3],
    ])

    aca = MatrixACA.from_matrix(A, (0, 0))

    assert aca.ncols() == 3
    assert aca.nrows() == 3
    assert aca.npivots() == 1
    assert aca.rowindices == [0]
    assert aca.colindices == [0]

    assert np.isclose(aca.evaluate(0, 0), A[0, 0])
    assert np.isclose(aca[0, 0], A[0, 0])
    assert np.allclose(aca.row(0), A[0, :])
    assert np.allclose(aca[0, :], A[0, :])
    assert np.allclose(aca.col(0), A[:, 0])
    assert np.allclose(aca[:, 0], A[:, 0])

    aca.add_pivot(A, (1, 2))

    assert aca.npivots() == 2
    assert aca.rowindices == [0, 1]
    assert aca.colindices == [0, 2]

    assert np.isclose(aca[1, 2], A[1, 2])
    assert np.isclose(aca.evaluate(1, 2), A[1, 2])
    assert np.allclose(aca[[0, 1], [0, 2]], A[np.ix_([0, 1], [0, 2])])
    assert np.allclose(aca.submatrix([0, 1], [0, 2]), A[np.ix_([0, 1], [0, 2])])

    aca.add_pivot(A)

    assert aca.npivots() == 3
    assert aca.rowindices == [0, 1, 2]
    assert aca.colindices == [0, 2, 1]

    assert np.allclose(aca.to_matrix(), A)
    assert np.allclose(aca[:, :], A)


def test_3x3_complex():
    A = np.array([
        [0.641325 + 0.331139j, 0.63414 + 0.902753j, 0.385012 + 0.359676j],
        [0.89194 + 0.783782j, 0.236955 + 0.0828438j, 0.98353 + 0.729723j],
        [0.219505 + 0.429946j, 0.544289 + 0.378888j, 0.14397 + 0.701327j],
    ])

    aca = MatrixACA.from_matrix(A, (0, 0))

    assert aca.ncols() == 3
    assert aca.nrows() == 3
    assert aca.npivots() == 1
    assert aca.rowindices == [0]
    assert aca.colindices == [0]

    aca.add_pivot(A)
    aca.add_pivot(A)

    assert np.allclose(aca.to_matrix(), A)
    assert np.allclose(aca[:, :], A)


def test_row_before_col_matches_col_before_row():
    """Adding a pivot's row first must give the same ACA as adding its column first.

    ``TensorCI1.add_global_pivot`` inserts every bond's pivot row before any bond's
    pivot column, so ``MatrixACA`` has to be order-agnostic. Upstream
    ``matrixaca.jl`` is not: it reads the pivot value off ``u[x_k, end]`` inside
    ``addpivotrow!``, which is the *previous* pivot's column when the row goes
    first, silently poisoning ``alpha`` and every later residual.
    """
    rng = np.random.default_rng(20240906)
    A = rng.random((8, 12))
    pivots = [(3, 5), (6, 1), (1, 9), (7, 4)]

    colfirst = MatrixACA.from_matrix(A, (0, 0))
    rowfirst = MatrixACA.from_matrix(A, (0, 0))
    for i, j in pivots:
        colfirst.add_pivot_col(A, j)
        colfirst.add_pivot_row(A, i)
        rowfirst.add_pivot_row(A, i)
        rowfirst.add_pivot_col(A, j)

        assert rowfirst.rowindices == colfirst.rowindices
        assert rowfirst.colindices == colfirst.colindices
        assert np.allclose(rowfirst.alpha, colfirst.alpha)
        assert np.allclose(rowfirst.to_matrix(), colfirst.to_matrix())

        # Cross-interpolation property: exact on the pivot rows and columns.
        residual = A - rowfirst.to_matrix()
        assert np.allclose(residual[rowfirst.rowindices, :], 0.0, atol=1e-12)
        assert np.allclose(residual[:, rowfirst.colindices], 0.0, atol=1e-12)


def test_rank_counts_complete_pivots_only():
    A = np.random.default_rng(7).random((5, 5))
    aca = MatrixACA.from_matrix(A, (0, 0))
    assert aca.rank() == 1

    aca.add_pivot_row(A, 2)  # half-added pivot: row present, column not yet
    assert aca.rank() == 1
    assert np.allclose(aca.to_matrix(), np.outer(A[:, 0], A[0, :]) / A[0, 0])

    aca.add_pivot_col(A, 3)
    assert aca.rank() == 2
