
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import median_filter
from sklearn.linear_model import LinearRegression
from matplotlib.widgets import EllipseSelector, Button

class BLUnmixer:
    def __init__(self, np_imstack=None):
        self.np_imstack = np_imstack
        self.rois = []
        self.K = None

    def get_ROIs(self):
        """
        This actually has to be run separately to work. Threaded implementation would
        be the solution to this in the future.
        """
        for image in self.np_imstack:
            self.rois.append(ROISelector(image))
    
#    def find_K(self):
        

from matplotlib.widgets import EllipseSelector, RectangleSelector
import numpy as np
import matplotlib.pyplot as plt

class ROISelector:
    def __init__(self, image):
        self.image = image
        self.coords = None
        self.isDone = False
        fig, ax = plt.subplots()
        self.widget = RectangleSelector(ax, self._onselect, drawtype='box',
                                             interactive=True,useblit=True)
        axDone = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.doneButton = Button(axDone, 'Done')
        self.doneButton.on_clicked(self.done)
        ax.imshow(self.image)
        plt.connect('key_press_event', self.widget)
        plt.show()
        
    def _onselect(self, pt1, pt2):
        #print(self.widget.geometry) # this will actually give the coords of each
        # point around the ellipse
        self.coords = np.array([[pt1.xdata, pt1.ydata],[pt2.xdata, pt2.ydata]])
        #print(self.coords)
    
    def done(self, event):
        self.isDone = True
        plt.close()
        #print('Done!')
        
    def get_coords(self):
        return self.coords