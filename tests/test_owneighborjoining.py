import random

import numpy as np
from AnyQt.QtTest import QSignalSpy
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from neighborjoining.owneighborjoining import OWNeighborJoining


class TestOWNeighborJoining(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.data = cls.data[random.sample(range(len(cls.data)), 10)]

        cls.distances = Euclidean(cls.data)
        cls.signal_name = "Distances"
        cls.signal_data = cls.distances
        cls.same_input_output_domain = True

    def setUp(self):
        self.widget = self.create_widget(OWNeighborJoining)

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 5)
        self.widget.select_indices(points)
        return sorted(points)

    def send_signal(self, input, value, *args, widget=None, wait=500):
        """ Send signal to widget by calling appropriate triggers.

        Parameters
        ----------
        input_name : str
        value : Object
        id : int
            channel id, used for inputs with flag Multiple
        widget : Optional[OWWidget]
            widget to send signal to. If not set, self.widget is used
        wait : int
            The amount of time to wait for the widget to complete.
        """
        if widget is None:
            widget = self.widget
        if isinstance(input, str):
            for input_signal in widget.get_signals("inputs"):
                if input_signal.name == input:
                    input = input_signal
                    break
            else:
                raise ValueError("'{}' is not an input name for widget {}"
                                 .format(input, type(widget).__name__))
        getattr(widget, input.handler)(value, *args)
        # the following part was added because set_distances needs time before handleNewSignals can send data to output
        # in the future handleNewDistances should probably be removed and set_distances should commit data
        if wait >= 0 and widget.isBlocking():
            spy = QSignalSpy(widget.blockingStateChanged)
            self.assertTrue(spy.wait(timeout=wait))

        widget.handleNewSignals()
        if wait >= 0 and widget.isBlocking():
            spy = QSignalSpy(widget.blockingStateChanged)
            self.assertTrue(spy.wait(timeout=wait))

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.distances, np.array([[0]]))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_all_zero_inputs(self):
        d = np.zeros((10, 10))
        self.widget.set_distances(d)

    def test_domain_loses_class(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)
        data = self.data[:, :4]
        distances = Euclidean(data)
        self.send_signal(self.widget.Inputs.distances, distances)
