#include "mainwindow.h"
#include "ui_mainwindow.h"

string last_item = "";
bool frame_updated = false;
QString dbName = QString::fromStdString(MODEL_PATH + "products.db");

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

// Executes when webcam start button pushed
void MainWindow::on_pushButton_open_webcam_clicked()
{
    // Update our shopping list based on what was selected in the product list
    ui->notifyLabel->setText("");
    for (int i = 0; i < ui->productsList->count(); i++) {
        auto item = ui->productsList->item(i);

        if (item->checkState() == Qt::Checked) {
            cout << "Adding " << item->text().toStdString() << endl;
            ui->shoppingList->addItem(item->text());
        }
    }

    // Ensure items were selected to shop.
    if (ui->shoppingList->count() == 0) {
        ui->notifyLabel->setText("<span style='font-size:14pt; font-weight:600; "
                                 "color:#aa0000;'>Please select items to shop for!</span>");
        return;
    }

    // Disable start
    ui->pushButton_open_webcam->setDisabled(true);
    ui->pushButton_open_webcam->setText("Shopping...");
    ui->productsList->setDisabled(true);
    ui->totalInputLabel->setText("$0.00");
    ui->shoppedList->clear();
    shopped.clear();

    // Open the camera
    cap.open(0);
    if(!cap.isOpened()){
        cout <<"camera is not open "<< endl;
        ui->pushButton_open_webcam->setDisabled(false);
        ui->pushButton_open_webcam->setText("Start");
        ui->productsList->setDisabled(false);
        return;
    }

    // Setup object detection
    ifstream ifs(string(string(MODEL_PATH) + class_file).c_str());
    string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }

    // load the neural network model
    model = readNet(MODEL_PATH + config_path, MODEL_PATH + weights_path);

    // Connect to our database
    m_db.setDatabaseName(dbName);
    if (!m_db.open()) {
        cout << "Error: connection with database failed" << endl;
    } else {
        cout << "Database: connection ok" << endl;
    }

    // Connect the timer's timeout signal to update window
    cout << "camera is open" << endl;
    connect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
    timer->start(20);
}

// Executes every time the timer times out
void MainWindow::update_window()
{
    // If our shopping list is empty, we're done!
    if (ui->shoppingList->count() == 0) {
        ui->notifyLabel->setText("<span style='font-size:14pt; font-weight:600; "
                                 "color:#00aa00;'>Done shopping!</span>");
        ui->nameInputLabel->setText("");
        ui->descriptionInputLabel->setText("");
        ui->priceInputLabel->setText("");
        ui->locationInputLabel->setText("");
        ui->pushButton_open_webcam->setDisabled(false);
        ui->pushButton_open_webcam->setText("Start");
        ui->productsList->setDisabled(false);
        cart_total = 0;
        cap.release();
        string ros_cmd = "cd " + MODEL_PATH + " && ./rosrun_wrapper.sh 9.00 1.00 1.00";
        system(ros_cmd.c_str());
        return;
    }

    // Update our name, description, etc. of active item
    auto active_item = ui->shoppingList->item(0);
    string active_item_name = active_item->text().toStdString();
    
    // Query for this item in the database
    QSqlQuery query;
    query.prepare("SELECT * FROM products WHERE name = (:name)");
    query.bindValue(":name", QString::fromStdString(active_item_name));
    if (!query.exec())
        cout << "ERROR: " << query.lastError().text().toStdString() << endl;

    // Update the UI with data about the active item
    QVariant item_name;
    QVariant item_desc;
    QVariant item_price;
    QVariant item_x;
    QVariant item_y;
    QVariant item_w;
    if (query.first()) {
        item_name = query.value(1);
        item_desc = query.value(2);
        item_price = query.value(3);
        item_x = query.value(4);
        item_y = query.value(5);
        item_w = query.value(6);
    }
    ui->nameInputLabel->setText(item_name.toString());
    ui->descriptionInputLabel->setText(item_desc.toString());
    ui->priceInputLabel->setText("$" + item_price.toString());
    ui->locationInputLabel->setText("(" + item_x.toString() + ", " + item_y.toString() + ")");

    if (frame_updated == true){
        string ros_cmd = "cd " + MODEL_PATH+ " && ./rosrun_wrapper.sh "
                        + item_x.toString().toStdString() + " " + item_y.toString().toStdString() + " 1.00";
        system(ros_cmd.c_str());
        frame_updated = false;
    } else if (last_item != active_item_name){
        last_item = active_item_name;
        frame_updated = true;
    }

    // Copy a frame from the camera
    cap >> frame;
    if (frame.empty())
        return; // end of video stream

    // Create blob from image
    Mat blob = blobFromImage(frame,
                             1.0 / 127.5,
                             Size(320, 320),
                             Scalar(127.5, 127.5, 127.5),
                             true,
                             false);
    model.setInput(blob);

    // Forward pass through the model to carry out the detection
    Mat output = model.forward();
    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // Loop through the detection material
    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);
        string name = class_names[class_id - 1];

        // Only detect things we care about
        if (std::find(std::begin(products), std::end(products), name) == std::end(products)) {
            continue;
        }

        // Check if the detection is of good quality
        if (confidence > 0.40) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);
            rectangle(frame,
                      Point(box_x, box_y),
                      Point(box_x + box_width, box_y + box_height),
                      Scalar(255, 255, 255),
                      2);
            putText(frame,
                    class_names[class_id - 1].c_str(),
                    Point(box_x, box_y - 5),
                    FONT_HERSHEY_SIMPLEX,
                    2,
                    Scalar(0, 255, 255),
                    2);

            // Skip items we've already shopped for
            if (std::find(std::begin(shopped), std::end(shopped), name) != std::end(shopped)) {
                continue;
            }

            // If we detected the active item, remove it from our items to buy list.
            if (name == item_name.toString().toStdString()) {
                // Update our shopped list and cart total
                cout << "got item " << name << "!" << endl;
                shopped.push_back(name);
                ui->shoppedList->addItem(item_name.toString());
                cart_total += item_price.toFloat();
                ui->totalInputLabel->setText("$" + QString::number(cart_total));
                ui->shoppingList->takeItem(0);
            }
        }
    }

    // Convert the OpenCV frame into a QImage
    cvtColor(frame, frame, COLOR_BGR2RGB);
    qt_image = QImage((const unsigned char *) (frame.data),
                      frame.cols,
                      frame.rows,
                      QImage::Format_RGB888);

    // Scale the image to the right size and update our label with the image.
    QImage smol = qt_image.scaled(640, 360);
    ui->label->setPixmap(QPixmap::fromImage(smol));
    ui->label->resize(640, 360);
}
