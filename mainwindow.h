#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <QMainWindow>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QTimer>

using namespace cv;
using namespace dnn;
using namespace std;

#define MODEL_PATH "/usr/share/opencv4/samples/data/dnn/"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_open_webcam_clicked();
    void update_window();


private:
    Ui::MainWindow *ui;
    QTimer *timer;

    // Database
    QSqlDatabase m_db = QSqlDatabase::addDatabase("QSQLITE");

    // Video Capture
    VideoCapture cap;
    Mat frame;
    QImage qt_image;

    // Detection
    float thres = 0.45; // Threshold to detect object
    vector<string> class_names;
    string class_file = "object_detection_classes_coco.txt"; // define file name with coco class names
    string config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    string weights_path = "frozen_inference_graph.pb";
    dnn4_v20211004::Net model;
    vector<string> products{"banana", "orange", "apple", "broccoli", "carrot"};

    // Cart
    float cart_total = 0;
    vector<string> shopped;
};
#endif // MAINWINDOW_H
