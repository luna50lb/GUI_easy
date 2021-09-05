from PyQt5 import QtCore, QtGui, QtWidgets

import pandas as pd
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import datetime 


class HTMLDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        painter.save()
        doc = QTextDocument()
        doc.setHtml(opt.text)
        opt.text = "";
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter)
        painter.translate(opt.rect.left(), opt.rect.top())
        clip = QRectF(0, 0, opt.rect.width(), opt.rect.height())
        doc.drawContents(painter, clip)
        painter.restore()

    def sizeHint(self, option, index ):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        doc = QTextDocument()
        doc.setHtml(opt.text);
        #doc.setTextWidth(opt.rect.width())
        idealWidth0=int(doc.idealWidth());
        size_and_height=int(doc.size().height())
        #print('ideal=', idealWidth0, 'size and height=', size_and_height)
        return QSize(idealWidth0, size_and_height) 

class QtDataFrame(QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        super().__init__();
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            value = value.toPyObject()
        else:
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

        
class LoginForm(QDialog):
    def __init__(self, parent=None):
        print('...')
        
class WindowD(QWidget):
    def __init__(self, parent=None):
        super().__init__();
        list_encodings=['Select Encoding', 'utf-8', 'cp932'] 
        hbox_outline=QHBoxLayout(self)
        hbox0=QHBoxLayout()
        hbox1=QHBoxLayout()
        hbox2=QHBoxLayout()
        #
        vbox0=QVBoxLayout()
        vbox1=QVBoxLayout()
        #
        vl0 = QVBoxLayout()
        holiz0=QHBoxLayout()
        holiz0.addWidget(QLabel('Latest'))
        holiz0.setAlignment(Qt.AlignCenter)
        hl0 = QHBoxLayout()
        self.combo0=QComboBox();
        self.combo0.addItems( list_encodings )
        hl0.addWidget(self.combo0)
        hl0.addWidget(QLabel(" .... ") )
        hl0.addWidget(QLabel(" **** ") )
        hl0.setAlignment(Qt.AlignLeft)
        #
        #self.loadBtn0 = QPushButton("Select File", self)
        #hl0.addWidget(self.loadBtn0)
        #self.pathLE0 = QLineEdit(self)
        #hl0.addWidget(self.pathLE0)
        vl0.addLayout(holiz0)
        vl0.addLayout(hl0)
        #self.pandasTv0 = QtWidgets.QTableView(self)
        self.pandasTv0 = QTableView(self)
        vl0.addWidget(self.pandasTv0)
        vl0.setAlignment(Qt.AlignBottom)
        #
        #
        vl1 = QVBoxLayout()
        holiz1=QHBoxLayout()
        holiz1.addWidget(QLabel('Big Data'))
        vl1.addLayout(holiz1);
        self.combo1=QComboBox();
        self.combo1.addItems( list_encodings )
        hl1 = QHBoxLayout()
        hl1.addWidget(self.combo1)
        self.loadBtn1 = QPushButton("Select File", self)
        hl1.addWidget(self.loadBtn1)
        self.pathLE1 = QLineEdit(self)
        hl1.addWidget(self.pathLE1)
        vl1.addLayout(hl1)
        self.pandasTv1 = QtWidgets.QTableView(self)
        vl1.addWidget(self.pandasTv1)
        #
        #
        hbox0.addLayout(vl0)
        hbox0.addLayout(vl1)
        #
        self.text_browser_0=QTextBrowser()
        self.text_browser_0.setText("horizontal line");
        #hbox1.addWidget(QLabel(" horiozontal line "));
        hbox1.addWidget(self.text_browser_0 )
        self.text_browser_1=QTextBrowser()
        self.text_browser_1.setText('app started at ' + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') )
        hbox2.addWidget(self.text_browser_1 )
        
        vbox0.addLayout(hbox0)
        #vbox0.addWidget(QLabel('*******'))
        vbox0.addLayout(hbox1)
        vbox0.addLayout(hbox2)
        #
        self.push_button_login=QPushButton('Login')
        vbox1.addWidget(self.push_button_login);
        vbox1.addWidget(QLabel('Automatic'))
        vbox1.addWidget(QPushButton('Start'))
        vbox1.addWidget(QLabel(' '))
        vbox1.addWidget(QLabel(' '))
        vbox1.addWidget(QLabel('Manual'))
        vbox1.addWidget(QPushButton('Deep Learning'))
        vbox1.addWidget(QPushButton('Send'))
        vbox1.setAlignment(Qt.AlignBottom) # PyQt5.QtCore.Qt.AlignBottom
        #
        hbox_outline.addLayout(vbox0)
        hbox_outline.addLayout(vbox1)
        #
        self.setWindowTitle('Main Window')
        self.push_button_login.clicked.connect(self.login)
        #self.loadBtn0.clicked.connect(self.loadFile0)
        self.pandasTv0.setItemDelegate(HTMLDelegate())
        self.pandasTv0.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.pandasTv0.setSortingEnabled(True)
        #
        self.loadBtn1.clicked.connect(self.loadFile1)
        self.pandasTv1.setItemDelegate(HTMLDelegate())
        self.pandasTv1.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.pandasTv1.setSortingEnabled(True)
    
    def login(self):
        print('login button clicked at', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        qdlg0=QDialog();
        frm_lyt=QFormLayout()
        line_edit_0=QLineEdit('Login');
        frm_lyt.addRow("line0", line_edit_0)
        line_edit_1=QLineEdit();
        frm_lyt.addRow("line1", line_edit_1)
        combo_0=QComboBox();
        combo_0.addItems( ['A', 'B', 'C', 'D'] )
        frm_lyt.addRow("Type:", combo_0)
        line_edit_username=QLineEdit();
        frm_lyt.addRow("username: ", line_edit_username)
        dialog_button_box0 = QDialogButtonBox(); # QtWidgets.QDialogButtonBox();
        dialog_button_box0.setOrientation(Qt.Horizontal)
        #dialog_button_box0.setStandardButtons(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel )
        dialog_button_box0.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply )
        #self.buttonBox.setObjectName("buttonBox")
        frm_lyt.addWidget(dialog_button_box0);
        qdlg0.setWindowTitle("Login Form")
        qdlg0.setLayout(frm_lyt)
        #
        dialog_button_box0.clicked.connect(qdlg0.accept)
        dialog_button_box0.rejected.connect(qdlg0.reject)
        #
        #qdlg0.exec_();
        if qdlg0.exec_()==QDialog.Accepted:
            dict_login={'line0':[line_edit_0.text()], 'line1':[line_edit_1.text()], 'combo0':[combo_0.currentText()],
                       'username':[line_edit_username.text() ] }
            self.df_login=pd.DataFrame(dict_login)
            print('df_login=\n', self.df_login)
        #print('qdlg0 exec_=', qdlg0.exec_(), ' qdialog Accepted', QDialog.Accepted)
        """ 
        def showdialog():
            dlg = QDialog()
            b1 = QPushButton("ok",dlg)
            b1.move(50,50)
            dlg.setWindowTitle("Dialog") 
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.exec_()
        """
        
        
    def loadFile0(self):
        print('check encoding!')
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",  "CSV Files (*csv)");
        #fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "",  "CSV Files (*csv)");
        self.pathLE0.setText(fileName)
        # error! encoding0=input('encoding=')
        encoding0='utf-8' # 'cp932'
        encoding0=self.combo0.currentText();
        if fileName.endswith('.csv'):
            df = pd.read_csv(fileName, encoding=encoding0, header=0)
        print('df shape, encoding=', df.shape, encoding0)
        model0 = QtDataFrame(df)
        self.pandasTv0.setModel(model0)

    def loadFile1(self):
        print('check encoding!')
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",  "CSV Files (*csv)");
        #fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "",  "CSV Files (*csv)");
        self.pathLE1.setText(fileName)
        encoding0='utf-8' # 'cp932'
        encoding0=self.combo1.currentText();
        if fileName.endswith('.csv'):
            df = pd.read_csv(fileName, encoding=encoding0, header=0)
        print('df shape, encoding=', df.shape, encoding0)
        model1 = QtDataFrame(df)
        self.pandasTv1.setModel(model1)            

if __name__ == "__main__":
    w0_width=1500;
    w0_height=750;
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    w0 = WindowD()
    w0.resize(w0_width, w0_height)
    w0.show()
    #sys.exit(app.exec_())
    app.exec();
