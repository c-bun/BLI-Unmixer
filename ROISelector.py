
from matplotlib.widgets import EllipseSelector
import numpy as np
import matplotlib.pyplot as plt

class ROISelector:
    def __init__(self, image):
        self.image = image
        self.coords = None
        fig, ax = plt.subplots()
        self.widget = EllipseSelector(ax, self._onselect, drawtype='line',
                                             interactive=True,useblit=True)
        plt.imshow(self.image)
        plt.connect('key_press_event', self.widget)
        plt.show()
        
    def _onselect(self, pt1, pt2):
        #print(self.widget.geometry)
        self.coords = np.array([pt1,pt2])
        print(self.coords)
        
    def get_coords(self):
        return self.coords