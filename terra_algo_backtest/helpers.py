import matplotlib.pyplot as plt


def pltShowWaitKey():
    plt.get_current_fig_manager().full_screen_toggle()
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
