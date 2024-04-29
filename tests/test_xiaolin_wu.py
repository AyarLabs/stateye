def test_xiaolin_wu():
    from stateye.xiaolin_wu import draw_line
    import numpy as np
    import math

    # Test steep sum
    hist = np.zeros(
        shape=(10, 10, 1),
        dtype=np.double,
    )
    x0, y0, x1, y1 = 1.5, 1.5, 5.0, 10.0
    draw_line(x0, y0, x1, y1, hist, nx=10, ny=10, zi=0)
    assert math.isclose(np.sum(hist), abs(y1 - y0))

    # Test shallow sum
    hist = np.zeros(
        shape=(10, 10, 1),
        dtype=np.double,
    )
    x0, y0, x1, y1 = 0, 0, 4.0, 2.0
    draw_line(x0, y0, x1, y1, hist, nx=10, ny=10, zi=0)
    assert math.isclose(np.sum(hist), abs(x1 - x0))

    # Plotting below
    # import matplotlib.pyplot as plt
    # plt.imshow(hist.T[::-1], extent=(-0.5, 9.5, -0.5, 9.5))
    # plt.plot([x0,x1],[y0,y1], 'r-')
    # plt.show()
