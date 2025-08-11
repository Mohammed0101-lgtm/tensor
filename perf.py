import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os

CSV_FILE = 'multiplication_test_results.csv'

fig, ax = plt.subplots()
line1, = ax.plot([], [], label='My Tensor Library')
line2, = ax.plot([], [], label='Eigen Tensor Library')
line3, = ax.plot([], [], label='XTensor Library')

ax.set_title('Real-time Performance Comparison')
ax.set_xlabel('Tensor Size')
ax.set_ylabel('Time (µs)')
ax.legend()
ax.grid(True)

def init():
    ax.set_xlim(0, 1000)   # Will auto-scale later
    ax.set_ylim(0, 1000)
    return line1, line2, line3

def update(frame):
    if not os.path.exists(CSV_FILE):
        return line1, line2, line3

    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        return line1, line2, line3

    if df.shape[0] < 2:
        return line1, line2, line3

    x = df['size']
    y1 = df['my_library']
    y2 = df['eigen_library']
    y3 = df['x_tensor_library']

    ax.clear()
    ax.set_title('Real-time Performance Comparison')
    ax.set_xlabel('Tensor Size')
    ax.set_ylabel('Time (µs)')
    ax.grid(True)

    ax.plot(x, y1, label='My Library')
    ax.plot(x, y2, label='Eigen Library')
    ax.plot(x, y3, label='XTensor Library')
    ax.legend()

    ax.relim()
    ax.autoscale_view()
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, init_func=init, interval=1000)
plt.tight_layout()
plt.show()