"""
Neighbor joining widget
------------------------

"""
import random
from collections import namedtuple
from itertools import chain
from math import atan2, pi
from types import SimpleNamespace as namespace
from xml.sax.saxutils import escape

import Orange
import numpy as np
import pkg_resources
import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem
from AnyQt.QtCore import Qt, QObject, QEvent, QSize, QRectF, QPointF
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtGui import (
    QColor, QPen, QBrush, QKeySequence, QPainterPath, QPainter, QCursor, QIcon
)
from AnyQt.QtWidgets import (
    QSlider, QToolButton, QFormLayout, QHBoxLayout,
    QSizePolicy, QAction, QActionGroup, QGraphicsPathItem,
    QGraphicsRectItem, QPinchGesture, QApplication
)
from Orange.canvas import report
from Orange.data import Table, DiscreteVariable, ContinuousVariable
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui, settings
from Orange.widgets.gui import ProgressBar
from Orange.widgets.utils import colorpalette
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.concurrent import ThreadExecutor
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owscatterplotgraph import LegendItem, legend_anchor_pos
from neighborjoining.neighbor_joining import (
    run_neighbor_joining, make_rooted, get_points_radial, get_points_circular, get_children, set_distance_floor
)


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


class TextItem(pg.TextItem):
    if not hasattr(pg.TextItem, "setAnchor"):
        # Compatibility with pyqtgraph <= 0.9.10; in (as of yet unreleased)
        # 0.9.11 the TextItem has a `setAnchor`, but not `updateText`
        def setAnchor(self, anchor):
            self.anchor = pg.Point(anchor)
            self.updateText()


class LegendItem(LegendItem):
    def __init__(self):
        super().__init__()
        self.items = []

    def clear(self):
        """
        Clear all legend items.
        """
        items = list(self.items)
        self.items = []
        for sample, label in items:
            # yes, the LegendItem shadows QGraphicsWidget.layout() with
            # an instance attribute.
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
            sample.hide()
            label.hide()

        self.updateSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            event.accept()
            if self.parentItem() is not None:
                self.autoAnchor(
                    self.pos() + (event.pos() - event.lastPos()) / 2)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()


Algorithm = namedtuple("Algorithm", ["name", "function"])
DRAWING_ALGORITHMS = (
    Algorithm("radial", get_points_radial),
    Algorithm("circular", get_points_circular)
)


class OWNeighborJoining(widget.OWWidget):
    name = "Neighbor Joining"
    description = "Display a phylogram constructed with neighbor joining " \
                  "from the inputted distance matrix."
    priority = 100

    inputs = [("Distances", DistMatrix, "set_distances")]
    outputs = [("Selected Data", Table, widget.Default),
               (ANNOTATED_DATA_SIGNAL_NAME, Table)]

    settingsHandler = settings.DomainContextHandler()

    attr_label = settings.ContextSetting(None, settings.ContextSetting.OPTIONAL, exclude_metas=False)
    attr_color = settings.ContextSetting(None, settings.ContextSetting.OPTIONAL, exclude_metas=False)
    attr_shape = settings.ContextSetting(None, settings.ContextSetting.OPTIONAL, exclude_metas=False)
    attr_size = settings.ContextSetting(None, settings.ContextSetting.OPTIONAL, exclude_metas=False)

    point_size = settings.Setting(10)
    alpha_value = settings.Setting(255)
    drawing_setting = settings.Setting(0)
    label_only_selected = settings.Setting(0)
    show_legend = settings.Setting(1)

    resolution = 256

    auto_commit = settings.Setting(True)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))
    MinPointSize = 6
    new_node_size_mult = 0.2

    ReplotRequest = QEvent.registerEventType()

    graph_name = "viewbox"

    def __init__(self):
        super().__init__()

        self.min_dist = 1e3 * np.finfo(float).eps
        self.root = 0
        self.tree = None
        self.rooted_tree = None

        self.matrix = None
        self.real = None
        self.new = None
        self.coords = None
        self._selection_mask = None
        self._item = None
        self.__legend = None
        self.__replot_requested = False
        self.__executor = ThreadExecutor()

        box = gui.vBox(self.controlArea, "Drawing")
        box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.label_model = DomainModel(placeholder="(No labels)")
        self.color_model = DomainModel(placeholder="(Same color)", valid_types=DomainModel.PRIMITIVE)
        self.shape_model = DomainModel(placeholder="(Same shape)", valid_types=DiscreteVariable)
        self.size_model = DomainModel(placeholder="(Same size)", valid_types=ContinuousVariable)

        common_options = dict(
            labelWidth=55, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str)

        def on_drawing_change():
            if self.matrix is None:
                return
            self.calculate_points()
            self._setup_plot()

        gui.comboBox(box, self, "drawing_setting",
                     callback=on_drawing_change,
                     items=tuple(alg.name for alg in DRAWING_ALGORITHMS),
                     label="Drawing:",
                     labelWidth=55,
                     orientation=Qt.Horizontal)

        box = gui.vBox(self.controlArea, "Points")

        gui.comboBox(box, self, "attr_color",
                     callback=self._on_color_change,
                     model=self.color_model,
                     label="Color:",
                     **common_options)
        gui.comboBox(box, self, "attr_label",
                     callback=self._on_label_change,
                     model=self.label_model,
                     label="Label:",
                     **common_options)
        gui.comboBox(box, self, "attr_shape",
                     callback=self._on_shape_change,
                     model=self.shape_model,
                     label="Shape:",
                     **common_options)
        gui.comboBox(box, self, "attr_size",
                     callback=self._on_size_change,
                     model=self.size_model,
                     label="Size:",
                     **common_options)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            spacing=8
        )
        box.layout().addLayout(form)

        size_slider = QSlider(
            Qt.Horizontal, minimum=3, maximum=30, value=self.point_size,
            pageStep=3,
            tickPosition=QSlider.TicksBelow)
        size_slider.valueChanged.connect(self._set_size)
        form.addRow("Symbol size:", size_slider)

        alpha_slider = QSlider(
            Qt.Horizontal, minimum=10, maximum=255, pageStep=25,
            tickPosition=QSlider.TicksBelow, value=self.alpha_value)
        alpha_slider.valueChanged.connect(self._set_alpha)
        form.addRow("Opacity:", alpha_slider)

        box = gui.vBox(self.controlArea, "Plot Properties")

        gui.checkBox(box, self, "show_legend", "Show legend", callback=self._update_legend)
        gui.checkBox(box, self, "label_only_selected", "Label only selected points", callback=self._on_label_change)

        toolbox = gui.vBox(self.controlArea, "Zoom/Select")
        toollayout = QHBoxLayout()
        toolbox.layout().addLayout(toollayout)

        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection",
                        auto_label="Send Automatically")

        self.controlArea.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Main area plot
        self.view = pg.GraphicsView(background="w")
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setFrameStyle(pg.GraphicsView.StyledPanel)
        self.viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        self.viewbox.setAspectLocked(True)
        self.viewbox.grabGesture(Qt.PinchGesture)
        self.view.setCentralItem(self.viewbox)

        self.mainArea.layout().addWidget(self.view)

        self.selection = PlotSelectionTool(self)
        self.selection.setViewBox(self.viewbox)
        self.selection.selectionFinished.connect(self._selection_finish)

        self.zoomtool = PlotZoomTool(self)
        self.pantool = PlotPanTool(self)
        self.pinchtool = PlotPinchZoomTool(self)
        self.pinchtool.setViewBox(self.viewbox)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(widget.__name__, path)
            return QIcon(path)

        actions = namespace(
            zoomtofit=QAction(
                "Zoom to fit", self, icon=icon("zoom_reset"),
                shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
                triggered=lambda: self.reset_view()),
            zoomin=QAction(
                "Zoom in", self,
                shortcut=QKeySequence(QKeySequence.ZoomIn),
                triggered=lambda: self.viewbox.scaleBy((1 / 1.25, 1 / 1.25))),
            zoomout=QAction(
                "Zoom out", self,
                shortcut=QKeySequence(QKeySequence.ZoomOut),
                triggered=lambda: self.viewbox.scaleBy((1.25, 1.25))),
            select=QAction(
                "Select", self, checkable=True, icon=icon("arrow"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_1)),
            zoom=QAction(
                "Zoom", self, checkable=True, icon=icon("zoom"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_2)),
            pan=QAction(
                "Pan", self, checkable=True, icon=icon("pan_hand"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_3)),
        )
        self.addActions([actions.zoomtofit, actions.zoomin, actions.zoomout])

        group = QActionGroup(self, exclusive=True)
        group.addAction(actions.select)
        group.addAction(actions.zoom)
        group.addAction(actions.pan)

        actions.select.setChecked(True)

        currenttool = self.selection

        def activated(action):
            nonlocal currenttool
            if action is actions.select:
                tool, cursor = self.selection, Qt.ArrowCursor
            elif action is actions.zoom:
                tool, cursor = self.zoomtool, Qt.ArrowCursor
            elif action is actions.pan:
                tool, cursor = self.pantool, Qt.OpenHandCursor
            else:
                assert False
            currenttool.setViewBox(None)
            tool.setViewBox(self.viewbox)
            self.viewbox.setCursor(QCursor(cursor))
            currenttool = tool

        group.triggered[QAction].connect(activated)

        def button(action):
            b = QToolButton()
            b.setDefaultAction(action)
            return b

        toollayout.addWidget(button(actions.select))
        toollayout.addWidget(button(actions.zoom))
        toollayout.addWidget(button(actions.pan))

        toollayout.addSpacing(4)
        toollayout.addWidget(button(actions.zoomtofit))
        toollayout.addStretch()
        toolbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def sizeHint(self):
        return QSize(800, 500)

    def clear(self):
        self.coords = None
        self._selection_mask = None

        self.attr_label = None
        self.attr_color = None
        self.attr_size = None
        self.attr_shape = None

        self.clear_plot()

    def clear_item(self):
        if self._item is not None:
            self._item.setParentItem(None)
            self.viewbox.removeItem(self._item)
            self._item = None

    def clear_legend(self):
        if self.__legend is not None:
            anchor = legend_anchor_pos(self.__legend)
            if anchor is not None:
                self.legend_anchor = anchor

            self.__legend.setParentItem(None)
            self.__legend.clear()
            self.__legend.setVisible(False)

    def clear_plot(self):
        self.clear_item()
        self.clear_legend()
        self.viewbox.clear()

    def _invalidate_plot(self):
        """
        Schedule a delayed replot.
        """
        if not self.__replot_requested:
            self.__replot_requested = True
            QApplication.postEvent(self, QEvent(self.ReplotRequest),
                                   Qt.LowEventPriority - 10)

    def calculate_points(self):
        """
        Calculate point coordinates and store them in self.coords. Uses distance matrix
        stored in self.matrix. The coordinates are stored in a 2D numpy array.
        """
        if self.matrix is None:
            return
        points = DRAWING_ALGORITHMS[self.drawing_setting].function(self.rooted_tree, self.root)
        self.real = np.arange(self.matrix.shape[0])
        self.new = np.arange(self.real[-1] + 1, len(points))
        self.coords = np.array([points[ix] for ix in sorted(points.keys())])

    def set_distances(self, matrix):
        self.closeContext()
        self.clear()
        self.information()

        self.__executor.submit(self.process_data, matrix)

    def process_data(self, matrix):
        self.setBlocking(True)
        self.setStatusMessage("Running")

        if matrix is not None and len(matrix) > 500:
            indices = random.sample(range(len(matrix)), 500)
            self.matrix = matrix.submatrix(indices)
            self.information("Only 500 records displayed")
        else:
            self.matrix = matrix

        if self.matrix is not None:
            progress = ProgressBar(self, len(self.matrix) - 2)
            self.tree = run_neighbor_joining(self.matrix, progress)
            self.root = len(self.tree) - 1
            self.rooted_tree = make_rooted(self.tree, self.root)
            set_distance_floor(self.rooted_tree, self.min_dist)

            self.calculate_points()

            if self.coords is not None and len(self.coords):
                self._initialize()
                self.openContext(self.matrix.row_items.domain)

                def findvar(name, iterable):
                    """Find a Orange.data.Variable in `iterable` by name"""
                    for el in iterable:
                        if isinstance(el, Orange.data.Variable) and el.name == name:
                            return el
                    else:
                        return None

                if isinstance(self.attr_label, str):
                    self.attr_label = findvar(self.attr_label, self.label_model)
                if isinstance(self.attr_color, str):
                    self.attr_color = findvar(self.attr_color, self.color_model)
                if isinstance(self.attr_shape, str):
                    self.attr_shape = findvar(self.attr_shape, self.shape_model)
                if isinstance(self.attr_size, str):
                    self.attr_size = findvar(self.attr_size, self.size_model)

                self._invalidate_plot()
            progress.finish()
        self.setBlocking(False)
        self.setStatusMessage("")

    def handleNewSignals(self):
        self.commit()

    def customEvent(self, event):
        if event.type() == OWNeighborJoining.ReplotRequest:
            self.__replot_requested = False
            self._setup_plot()
        else:
            super().customEvent(event)

    def _initialize(self):
        # Initialize the GUI controls.
        domain = self.matrix.row_items and self.matrix.row_items.domain
        self.label_model.set_domain(domain)
        self.color_model.set_domain(domain)
        self.size_model.set_domain(domain)
        self.shape_model.set_domain(domain)

        self.attr_label = None
        self.attr_color = domain and self.matrix.row_items.domain.class_var or None
        self.attr_size = None
        self.attr_shape = None

    def _get_data(self, var, dtype):
        """
        Return the column data and mask for variable `var`
        """
        return column_data(self.matrix.row_items, var, dtype)

    def reset_view(self):
        self.viewbox.setRange(QRectF(-1, -1, 2, 2))
        if not self.label_only_selected:
            additional = max(item.boundingRect().width() for item in self._labels)*max(self.viewbox.viewPixelSize())#/500
            self.viewbox.setRange(QRectF(-1-additional, -1-additional, 2+2*additional, 2+2*additional))

    def _setup_plot(self, reset_view=True):
        self.__replot_requested = False
        self.clear_plot()

        X, Y = self.coords.T
        X -= (X.max() + X.min()) / 2
        Y -= (Y.max() + Y.min()) / 2
        maxspan = max(X.max() - X.min(), Y.max() - Y.min())
        X *= 2 / maxspan
        Y *= 2 / maxspan

        pen_data, brush_data = self._color_data()
        size_data = self._size_data()
        shape_data = self._shape_data()

        if self.rooted_tree is not None:
            l = list(chain.from_iterable((ix1, ix2) for ix1 in self.rooted_tree for ix2, _ in self.rooted_tree[ix1]))
            lines = pg.PlotDataItem(X[l],Y[l], connect="pairs")
            self.viewbox.addItem(lines)

        if size_data.shape == self.real.shape:
            size_data = np.hstack((size_data, np.full(self.new.shape, self.point_size * self.new_node_size_mult)))
        else:
            size_data[self.new] = self.point_size * self.new_node_size_mult
        self._item = ScatterPlotItem(
            X, Y,
            pen=pen_data,
            brush=brush_data,
            size=size_data,
            symbol=shape_data,
            antialias=True,
            data=np.arange(len(self.coords))
        )

        self.viewbox.addItem(self._item)

        self._labels = []
        if DRAWING_ALGORITHMS[self.drawing_setting].name == "radial":
            angles = np.empty(self.real.shape)
            for v1 in self.rooted_tree:
                for v2 in get_children(self.rooted_tree, v1):
                    if v2 < len(self.real):
                        delta = self.coords[v2, :2] - self.coords[v1, :2]
                        angles[v2] = atan2(delta[1], delta[0])
        for i in self.real:
            if DRAWING_ALGORITHMS[self.drawing_setting].name == "circular":
                angle = atan2(Y[i], X[i])
            elif DRAWING_ALGORITHMS[self.drawing_setting].name == "radial":
                angle = angles[i]

            if pi/2 < (angle + 2*pi) % (2*pi) < 3*pi/2:
                angle = angle + pi
                anchor = (1, 0.5)
            else:
                anchor = (0, 0.5)

            item = pg.TextItem("", anchor=anchor, angle=angle*180/pi, color=(0, 0, 0))
            item.setPos(X[i], Y[i])
            self.viewbox.addItem(item)
            self._labels.append(item)

        self.set_label(self._label_data(mask=None))

        if reset_view:
            self.reset_view()
        self._update_legend()

    def _label_data(self, mask=None):
        label_var = self.attr_label
        if label_var is None:
            label_data = np.full(self.real.shape, "", dtype=str)
        else:
            if label_var.is_string:
                label_data, label_mask = self._get_data(label_var, dtype=str)
            else:
                label_data, label_mask = self._get_data(label_var, dtype=float)
                label_data = np.array(list(map(label_var.repr_val, label_data)))
            label_data[label_mask] = ""
        if mask is None:
            return label_data
        else:
            return label_data[mask]

    def _on_label_change(self):
        if self.coords is None:
            return
        self.set_label(self._label_data(mask=None))

        vr = self.viewbox.viewRange()
        if not self.label_only_selected and vr[0][0] == -vr[0][1] and vr[1][0] == -vr[1][1]:
            self.reset_view()

    def _color_data(self, mask=None):
        color_var = self.attr_color
        if color_var is not None:
            color_data, _ = self._get_data(color_var, dtype=float)
            if color_var.is_continuous:
                color_data = plotutils.continuous_colors(
                    color_data, None, *color_var.colors)
            else:
                color_data = plotutils.discrete_colors(
                    color_data, len(color_var.values),
                    color_index=color_var.colors
                )
            color_data = np.vstack((color_data, [QColor(Qt.darkGray).getRgb()[:3]] * len(self.new)))
            if mask is not None:
                color_data = color_data[mask]

            pen_data = np.array(
                [pg.mkPen((r, g, b), width=1.5)
                 for r, g, b in color_data * 0.8],
                dtype=object)

            brush_data = np.array(
                [pg.mkBrush((r, g, b, self.alpha_value))
                 for r, g, b in color_data],
                dtype=object)
        else:
            color = QColor(Qt.darkGray)
            pen_data = QPen(color, 1.5)
            pen_data.setCosmetic(True)
            color = QColor(Qt.lightGray)
            color.setAlpha(self.alpha_value)
            brush_data = QBrush(color)

        if self._selection_mask is not None:
            assert self._selection_mask.shape == (len(self.coords),)
            if mask is not None:
                selection_mask = self._selection_mask[mask]
            else:
                selection_mask = self._selection_mask

            if isinstance(pen_data, QPen):
                pen_data = np.array([pen_data] * selection_mask.size,
                                       dtype=object)

            pen_data[selection_mask] = pg.mkPen((255, 190, 0, 255), width=4)
        return pen_data, brush_data

    def _on_color_change(self):
        if self.coords is None or self._item is None:
            return

        pen, brush = self._color_data()

        if isinstance(pen, QPen):
            # Reset the brush for all points
            self._item.data["pen"] = None
            self._item.setPen(pen)
        else:
            self._item.setPen(pen)

        if isinstance(brush, QBrush):
            # Reset the brush for all points
            self._item.data["brush"] = None
            self._item.setBrush(brush)
        else:
            self._item.setBrush(brush)

        self._update_legend()

    def _shape_data(self, mask=None):
        shape_var = self.attr_shape
        if shape_var is None:
            shape_data = np.array(["o"] * len(self.coords))
        else:
            assert shape_var.is_discrete
            symbols = np.array(list(ScatterPlotItem.Symbols))
            max_symbol = symbols.size - 1
            shapeidx, shape_mask = column_data(self.matrix.row_items, shape_var, dtype=int)
            shapeidx = np.hstack((shapeidx, np.full(self.new.shape, -1)))
            shape_mask = np.hstack((shape_mask, np.full(self.new.shape, False)))
            shapeidx[shape_mask] = max_symbol
            shapeidx[~shape_mask] %= max_symbol -1
            shape_data = symbols[shapeidx]
        if mask is None:
            return shape_data
        else:
            return shape_data[mask]

    def _on_shape_change(self):
        if self.coords is None:
            return

        self.set_shape(self._shape_data(mask=None))
        self._update_legend()

    def _size_data(self, mask=None):
        size_var = self.attr_size
        if size_var is None:
            size_data = np.full((len(self.coords),), self.point_size,
                                dtype=float)
        else:
            nan_size = OWNeighborJoining.MinPointSize - 2
            size_data, size_mask = self._get_data(size_var, dtype=float)
            size_data_valid = size_data[~size_mask]
            if size_data_valid.size:
                smin, smax = np.min(size_data_valid), np.max(size_data_valid)
                sspan = smax - smin
            else:
                sspan = smax = smin = 0
            size_data[~size_mask] -= smin
            if sspan > 0:
                size_data[~size_mask] /= sspan
            size_data = \
                size_data * self.point_size + OWNeighborJoining.MinPointSize
            size_data[size_mask] = nan_size
        if mask is None:
            return size_data
        else:
            return size_data[mask]

    def _on_size_change(self):
        if self.coords is None:
            return
        self.set_size(self._size_data(mask=None))
        self.update_label_offset()

    def _update_legend(self):
        if self.__legend is None:
            self.__legend = legend = LegendItem()
            legend.setParentItem(self.viewbox)
            legend.setZValue(self.viewbox.zValue() + 10)
            legend.restoreAnchor(self.legend_anchor)
        else:
            legend = self.__legend

        legend.clear()

        color_var, shape_var = self.attr_color, self.attr_shape
        if color_var is not None and not color_var.is_discrete:
            color_var = None
        assert shape_var is None or shape_var.is_discrete
        if not self.show_legend or color_var is None and shape_var is None:
            legend.setParentItem(None)
            legend.hide()
            return
        else:
            if legend.parentItem() is None:
                legend.setParentItem(self.viewbox)
            legend.setVisible(True)

        symbols = list(ScatterPlotItem.Symbols)

        if shape_var is color_var:
            items = [(QColor(*color_var.colors[i]), symbols[i], name)
                     for i, name in enumerate(color_var.values)]
        else:
            colors = shapes = []
            if color_var is not None:
                colors = [(QColor(*color_var.colors[i]), "o", name)
                          for i, name in enumerate(color_var.values)]
            if shape_var is not None:
                shapes = [(QColor(Qt.gray),
                           symbols[i % (len(symbols) - 1)], name)
                          for i, name in enumerate(shape_var.values)]
            items = colors + shapes

        for color, symbol, name in items:
            legend.addItem(
                ScatterPlotItem(pen=color, brush=color, symbol=symbol, size=10),
                escape(name)
            )

    def update_label_offset(self):
        """Update offsets of labels form points."""
        size_data = self._size_data()
        for i in self.real:
            item = self._labels[i]
            offset = size_data[i]/2
            if item.anchor[0] > 0:
                anchor = (1 + offset/item.boundingRect().width(), 0.5)
            else:
                anchor = (-offset/item.boundingRect().width(), 0.5)
            item.setAnchor(anchor)

    def set_label(self, labels):
        for i in self.real:
            if not self.label_only_selected or self._selection_mask is not None and self._selection_mask[i]:
                self._labels[i].setText(labels[i])
            else:
                self._labels[i].setText("")

        self.update_label_offset()

    def set_shape(self, shape):
        """
        Set (update) the current point shape map.
        """
        if self._item is not None:
            self._item.setSymbol(shape)

    def set_size(self, size):
        """
        Set (update) the current point size.
        """
        if self._item is not None:
            if len(size) <= self.new[0]:
                size = np.hstack((size, np.full(len(self.new), self.point_size * self.new_node_size_mult)))
            else:
                size[self.new] = self.point_size * self.new_node_size_mult
            self._item.setSize(size)

    def _set_alpha(self, value):
        self.alpha_value = value
        self._on_color_change()

    def _set_size(self, value):
        self.point_size = value
        self._on_size_change()

    def _selection_finish(self, path):
        self.select(path)

    def select(self, selectionshape):
        item = self._item
        if item is None:
            return

        indices = [spot.data()
                   for spot in item.points()
                   if selectionshape.contains(spot.pos()) and spot.data() < self.new[0]]

        self.select_indices(indices, QApplication.keyboardModifiers())

    def select_indices(self, indices, modifiers=Qt.NoModifier):
        if self.coords is None:
            return

        if self._selection_mask is None or \
                not modifiers & (Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier):
            self._selection_mask = np.zeros(len(self.coords), dtype=bool)

        if modifiers & Qt.AltModifier:
            self._selection_mask[indices] = False
        elif modifiers & Qt.ControlModifier:
            self._selection_mask[indices] = ~self._selection_mask[indices]
        else:
            self._selection_mask[indices] = True

        if self.label_only_selected:
            self._on_label_change()

        self._on_color_change()
        self.commit()

    def commit(self):
        subset = None
        indices = None
        if self.coords is not None and self._selection_mask is not None:
            indices = np.flatnonzero(self._selection_mask[:len(self.matrix.row_items)])
            if len(indices) > 0:
                subset = self.matrix.row_items[indices]

        self.send("Selected Data", subset)
        if self.matrix is not None:
            self.send(ANNOTATED_DATA_SIGNAL_NAME,
                      create_annotated_table(self.matrix.row_items, indices))
        else:
            self.send(ANNOTATED_DATA_SIGNAL_NAME, None)

    def send_report(self):
        self.report_plot(name="", plot=self.viewbox.getViewBox())
        def name(var):
            return var and var.name
        caption = report.render_items_vert((
            ("Color", name(self.attr_color)),
            ("Label", name(self.attr_label)),
            ("Shape", name(self.attr_shape)),
            ("Size", name(self.attr_size))
        ))
        self.report_caption(caption)


class PlotTool(QObject):
    """
    An abstract tool operating on a pg.ViewBox.

    Subclasses of `PlotTool` implement various actions responding to
    user input events. For instance `PlotZoomTool` when active allows
    the user to select/draw a rectangular region on a plot in which to
    zoom.

    The tool works by installing itself as an `eventFilter` on to the
    `pg.ViewBox` instance and dispatching events to the appropriate
    event handlers `mousePressEvent`, ...

    When subclassing note that the event handlers (`mousePressEvent`, ...)
    are actually event filters and need to return a boolean value
    indicating if the event was handled (filtered) and should not propagate
    further to the view box.

    See Also
    --------
    QObject.eventFilter

    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__viewbox = None

    def setViewBox(self, viewbox):
        """
        Set the view box to operate on.

        Call ``setViewBox(None)`` to remove the tool from the current
        view box. If an existing view box is already set it is first
        removed.

        .. note::
            The PlotTool will install itself as an event filter on the
            view box.

        Parameters
        ----------
        viewbox : pg.ViewBox or None

        """
        if self.__viewbox is viewbox:
            return
        if self.__viewbox is not None:
            self.__viewbox.removeEventFilter(self)
            self.__viewbox.destroyed.disconnect(self.__viewdestroyed)

        self.__viewbox = viewbox

        if self.__viewbox is not None:
            self.__viewbox.installEventFilter(self)
            self.__viewbox.destroyed.connect(self.__viewdestroyed)

    def viewBox(self):
        """
        Return the view box.

        Returns
        -------
        viewbox : pg.ViewBox
        """
        return self.__viewbox

    @Slot(QObject)
    def __viewdestroyed(self, _):
        self.__viewbox = None

    def mousePressEvent(self, event):
        """
        Handle a mouse press event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseMoveEvent(self, event):
        """
        Handle a mouse move event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseReleaseEvent(self, event):
        """
        Handle a mouse release event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseDoubleClickEvent(self, event):
        """
        Handle a mouse double click event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def gestureEvent(self, event):
        """
        Handle a gesture event.

        Parameters
        ----------
        event : QGraphicsSceneGestureEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def eventFilter(self, obj, event):
        """
        Reimplemented from `QObject.eventFilter`.
        """
        if obj is self.__viewbox:
            if event.type() == QEvent.GraphicsSceneMousePress:
                return self.mousePressEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseMove:
                return self.mouseMoveEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                return self.mouseReleaseEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseDoubleClick:
                return self.mouseDoubleClickEvent(event)
            elif event.type() == QEvent.Gesture:
                return self.gestureEvent(event)
        return super().eventFilter(obj, event)


class PlotSelectionTool(PlotTool):
    """
    A tool for selecting a region on a plot.

    """
    #: Selection modes
    Rect, Lasso = 1, 2

    #: Selection was started by the user.
    selectionStarted = Signal(QPainterPath)
    #: The current selection has been updated
    selectionUpdated = Signal(QPainterPath)
    #: The selection has finished (user has released the mouse button)
    selectionFinished = Signal(QPainterPath)

    def __init__(self, parent=None, selectionMode=Rect, **kwargs):
        super().__init__(parent, **kwargs)
        self.__mode = selectionMode
        self.__path = None
        self.__item = None

    def setSelectionMode(self, mode):
        """
        Set the selection mode (rectangular or lasso selection).

        Parameters
        ----------
        mode : int
            PlotSelectionTool.Rect or PlotSelectionTool.Lasso

        """
        assert mode in {PlotSelectionTool.Rect, PlotSelectionTool.Lasso}
        if self.__mode != mode:
            if self.__path is not None:
                self.selectionFinished.emit()
            self.__mode = mode
            self.__path = None

    def selectionMode(self):
        """
        Return the current selection mode.
        """
        return self.__mode

    def selectionShape(self):
        """
        Return the current selection shape.

        This is the area selected/drawn by the user.

        Returns
        -------
        shape : QPainterPath
            The selection shape in view coordinates.
        """
        if self.__path is not None:
            shape = QPainterPath(self.__path)
            shape.closeSubpath()
        else:
            shape = QPainterPath()
        viewbox = self.viewBox()

        if viewbox is None:
            return QPainterPath()

        return viewbox.childGroup.mapFromParent(shape)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(pos, pos)
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path = QPainterPath()
                self.__path.moveTo(event.pos())
            self.selectionStarted.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionUpdated.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionFinished.emit(self.selectionShape())
            self.__path = QPainterPath()
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return

        if self.__path.isEmpty():
            if self.__item is not None:
                self.__item.setParentItem(None)
                viewbox.removeItem(self.__item)
                if self.__item.scene():
                    self.__item.scene().removeItem(self.__item)
                self.__item = None
        else:
            if self.__item is None:
                item = QGraphicsPathItem()
                color = QColor(Qt.yellow)
                item.setPen(QPen(color, 0))
                color.setAlpha(50)
                item.setBrush(QBrush(color))
                self.__item = item
                viewbox.addItem(item)

            self.__item.setPath(self.selectionShape())


class PlotZoomTool(PlotTool):
    """
    A zoom tool.

    Allows the user to draw a rectangular region to zoom in.
    """

    zoomStarted = Signal(QRectF)
    zoomUpdated = Signal(QRectF)
    zoomFinished = Signal(QRectF)

    def __init__(self, parent=None, autoZoom=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__zoomrect = QRectF()
        self.__zoomitem = None
        self.__autozoom = autoZoom

    def zoomRect(self):
        """
        Return the current drawn rectangle (region of interest)

        Returns
        -------
        zoomrect : QRectF
        """
        view = self.viewBox()
        if view is None:
            return QRectF()
        return view.childGroup.mapRectFromParent(self.__zoomrect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(event.pos(), event.pos())
            self.zoomStarted.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()
            self.zoomUpdated.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.buttons() & Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()

            if self.__autozoom:
                PlotZoomTool.pushZoomRect(self.viewBox(), self.zoomRect())

            self.zoomFinished.emit(self.zoomRect())
            self.__zoomrect = QRectF()
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            PlotZoomTool.popZoomStack(self.viewBox())
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return
        if not self.__zoomrect.isValid():
            if self.__zoomitem is not None:
                self.__zoomitem.setParentItem(None)
                viewbox.removeItem(self.__zoomitem)
                if self.__zoomitem.scene() is not None:
                    self.__zoomitem.scene().removeItem(self.__zoomitem)
                self.__zoomitem = None
        else:
            if self.__zoomitem is None:
                self.__zoomitem = QGraphicsRectItem()
                color = QColor(Qt.yellow)
                self.__zoomitem.setPen(QPen(color, 0))
                color.setAlpha(50)
                self.__zoomitem.setBrush(QBrush(color))
                viewbox.addItem(self.__zoomitem)

            self.__zoomitem.setRect(self.zoomRect())

    @staticmethod
    def pushZoomRect(viewbox, rect):
        viewbox.showAxRect(rect)
        viewbox.axHistoryPointer += 1
        viewbox.axHistory[viewbox.axHistoryPointer:] = [rect]

    @staticmethod
    def popZoomStack(viewbox):
        if viewbox.axHistoryPointer == 0:
            viewbox.autoRange()
            viewbox.axHistory = []
            viewbox.axHistoryPointer = -1
        else:
            viewbox.scaleHistory(-1)


class PlotPanTool(PlotTool):
    """
    Pan/translate tool.
    """
    panStarted = Signal()
    translated = Signal(QPointF)
    panFinished = Signal()

    def __init__(self, parent=None, autoPan=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__autopan = autoPan

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panStarted.emit()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            viewbox = self.viewBox()
            delta = (viewbox.mapToView(event.pos()) -
                     viewbox.mapToView(event.lastPos()))
            if self.__autopan:
                viewbox.translateBy(-delta / 2)
            self.translated.emit(-delta / 2)
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panFinished.emit()
            event.accept()
            return True
        else:
            return False


class PlotPinchZoomTool(PlotTool):
    """
    A tool implementing a "Pinch to zoom".
    """

    def gestureEvent(self, event):
        gesture = event.gesture(Qt.PinchGesture)
        if gesture.state() == Qt.GestureStarted:
            event.accept(gesture)
            return True
        elif gesture.changeFlags() & QPinchGesture.ScaleFactorChanged:
            viewbox = self.viewBox()
            center = viewbox.mapSceneToView(gesture.centerPoint())
            scale_prev = gesture.lastScaleFactor()
            scale = gesture.scaleFactor()
            if scale_prev != 0:
                scale = scale / scale_prev
            if scale > 0:
                viewbox.scaleBy((1 / scale, 1 / scale), center)
            event.accept()
            return True
        elif gesture.state() == Qt.GestureFinished:
            viewbox = self.viewBox()
            PlotZoomTool.pushZoomRect(viewbox, viewbox.viewRect())
            event.accept()
            return True
        else:
            return False


def column_data(table, var, dtype):
    dtype = np.dtype(dtype)
    col, copy = table.get_column_view(var)
    if var.is_primitive() and not isinstance(col.dtype.type, np.inexact):
        # from mixes metas domain
        col = col.astype(float)
        copy = True
    if var.is_string:
        mask = (col == "")
    else:
        mask = np.isnan(col)
    if dtype != col.dtype:
        col = col.astype(dtype)
        copy = True

    if not copy:
        col = col.copy()
    return col, mask


class plotutils:
    @staticmethod
    def continuous_colors(data, palette=None,
                          low=(220, 220, 220), high=(0,0,0),
                          through_black=False):
        if palette is None:
            palette = colorpalette.ContinuousPaletteGenerator(
                QColor(*low), QColor(*high), through_black)
        amin, amax = np.nanmin(data), np.nanmax(data)
        span = amax - amin
        data = (data - amin) / (span or 1)
        return palette.getRGB(data)

    @staticmethod
    def discrete_colors(data, nvalues, palette=None, color_index=None):
        if color_index is None:
            if palette is None or nvalues >= palette.number_of_colors:
                palette = colorpalette.ColorPaletteGenerator(nvalues)
            color_index = palette.getRGB(np.arange(nvalues))
        # Unknown values as gray
        # TODO: This should already be a part of palette
        color_index = np.vstack((color_index, [[128, 128, 128]]))

        data = np.where(np.isnan(data), nvalues, data)
        data = data.astype(int)
        return color_index[data]

    @staticmethod
    def normalized(a):
        if not a.size:
            return a.copy()
        amin, amax = np.nanmin(a), np.nanmax(a)
        if np.isnan(amin):
            return a.copy()
        span = amax - amin
        mean = np.nanmean(a)
        return (a - mean) / (span or 1)


def test_main(argv=None):
    import sys
    import sip

    import Orange.distance as distance

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "yeast-class-RPR"

    data = Table(filename)
    matrix = distance.Euclidean(distance._preprocess(data))
    app = QApplication([])
    w = OWNeighborJoining()
    w.set_distances(matrix)
    w.handleNewSignals()
    w.show()
    w.raise_()
    r = app.exec()
    w.set_distances(None)
    w.saveSettings()
    sip.delete(w)
    del w
    return r


if __name__ == "__main__":
    import sys
    sys.exit(test_main())
