import org.opencv.core.*;
import org.opencv.dnn.DetectionModel;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.function.Consumer;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static org.opencv.core.CvType.CV_32F;
//import org.tensorflow.ConcreteFunction;
//import org.tensorflow.Signature;
//import org.tensorflow.Tensor;
//import org.tensorflow.TensorFlow;
//import org.tensorflow.op.Ops;
//import org.tensorflow.op.core.Placeholder;
//import org.tensorflow.op.math.Add;
//import org.tensorflow.types.TInt32;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // вывести в окне картинку
        // нарисовать рамки в области распознавания
        // запуск по ключу
        SwingUtilities.invokeLater(() -> {
            try {
                BufferedImage img_origin = ImageIO.read(new File(args[0]));
//                    BufferedImage resizedImage = res  izeImage(img, 640, 480);
                Image tmp = img_origin.getScaledInstance(640, 480, Image.SCALE_SMOOTH);
                BufferedImage dimg = new BufferedImage(640, 480, BufferedImage.TYPE_4BYTE_ABGR);
                Graphics2D g2d2 = dimg.createGraphics();
                g2d2.drawImage(tmp, 0, 0, null);
                g2d2.dispose();
                CatImageDetector detecor = new CatImageDetector(args[0], args[1]);
                detecor.work_with_image(dimg);
                if (args.length > 2 && args[2].equalsIgnoreCase("-g")) {
                    Graphics2D g2d = dimg.createGraphics();
                    g2d.setColor(Color.RED);
                    g2d.setStroke(new BasicStroke(5));

//                Rect[] array = detecor.faces.toArray();
//                for (Rect rect : array) {
//                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
//                }
                    byte[] pixels = ((DataBufferByte) dimg.getRaster().getDataBuffer()).getData();
                    Mat inputFrame = new Mat(dimg.getHeight(), dimg.getWidth(), CvType.CV_8UC3);
                    inputFrame.put(0, 0, pixels);
                    Mat mat = Dnn.blobFromImage(inputFrame, 1, new Size(640, 640));
//                    detecor.rezults.forEach(new Consumer<Mat>() {
//                        @Override
//                        public void accept(Mat mat) {
////                                                        mat.get()
//                        }
//                    });
//                    for (int i = 0; i < detecor.detect_rez.rows(); i++) {
////                    float confidence = detecor.detect_rez.at(Float.class, i, 2).getV();
//                        // Check if the detection is of good quality
//                        int x = (int) (detecor.detect_rez.at(Float.class, i, 3).getV() * mat.cols());
//                        int y = (int) (detecor.detect_rez.at(Float.class, i, 4).getV() * mat.rows());
//                        int width = (int) (detecor.detect_rez.at(Float.class, i, 5).getV() * mat.cols() - x);
//                        int height = (int) (detecor.detect_rez.at(Float.class, i, 6).getV() * mat.rows() - y);
//                        g2d.drawRect(x, y, width, height);
//
//                    }
//                g2d.setColor(Color.GREEN);
//                g2d.setStroke(new BasicStroke(5));
//                Rect[] array2 = detecor.faces2.toArray();
//                for (Rect rect : array2) {
//                    g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
//                }
                    g2d.dispose();
                    ImageIcon icon = new ImageIcon(dimg);
                    JFrame window = new JFrame("Результат");
                    window.setLocationByPlatform(true);
                    window.setPreferredSize(new Dimension(dimg.getWidth(), dimg.getHeight()));
                    window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                    JLabel lbl = new JLabel();
                    lbl.setIcon(icon);
                    window.add(lbl);
                    window.getContentPane().add(lbl, BorderLayout.CENTER);
                    window.pack();
                    window.setVisible(true);
                }

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        // использование TensorFlow
//        try (ConcreteFunction dbl = ConcreteFunction.create(Main::dbl);
//             TInt32 x = TInt32.scalarOf(10);
//             Tensor dblX = dbl.call(x)) {
//            System.out.println(x.getInt() + " doubled is " + ((TInt32)dblX).getInt());
//        }
    }

//    static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
//        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
//        Graphics2D graphics2D = resizedImage.createGraphics();
//        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
//        graphics2D.dispose();
//        return resizedImage;
//    }

//    private static Signature dbl(Ops tf) {
//        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
//        Add<TInt32> dblX = tf.math.add(x, x);
//        return Signature.builder().input("x", x).output("dbl", dblX).build();
//    }
}

class CatImageDetector {

    MatOfRect faces = new MatOfRect();
    MatOfRect faces2 = new MatOfRect();
    Mat detect_rez = new Mat();
    LinkedList<Mat> rezults;
    private final String model_weights;
    private final String model_config;
    private final String current_dir;
    private final String class_file_name_dir;
    private final String output_path;
    private final List<String> classes;
    private final List<String> output_layers;
    private String input_path;
    private List<String> layer_names;
    private Net network;
    private Size size;
    private Integer height;
    private Integer width;
    private Integer channels;
    private Scalar mean;
    private Mat image;
    private Mat blob;
    private List<Mat> outputs;
    private List<Rect2d> boxes;
    private List<Float> confidences;
    private List<Integer> class_ids;
    private String outputFileName;
    private boolean save;
    private boolean errors;
    String path_to_model;
    Mat mat;

    public CatImageDetector(String input_path, String path_to_model) {
        this.input_path = input_path;
        this.output_path = "";
        this.outputFileName = outputFileName;
        boxes = new ArrayList<>();
        classes = new ArrayList<>();
        class_ids = new ArrayList<>();
        layer_names = new ArrayList<>();
        confidences = new ArrayList<>();
        double[] means = {0.0, 0.0, 0.0};
        mean = new Scalar(means);
        output_layers = new ArrayList<>();
        size = new Size(640, 640);
        current_dir = System.getProperty("user.dir");
        model_weights = current_dir + "/Assets/models/yolov3-608.weights";
        model_config = current_dir + "/Assets/models/yolov3-608.cfg";
        class_file_name_dir = current_dir + "/Assets/models/coco.names";
        save = false;
        this.path_to_model = path_to_model;
    }

    public int[] work_with_image(BufferedImage img) {

        int[] rez = {0, 0};

//        mat = matify(img);
//        network = Dnn.readNetFromONNX(path_to_model);
        network = Dnn.readNet(path_to_model);
//        image = Dnn.blobFromImage(mat, 1, new Size(640, 640));
        // доделать нейросеть
        loadPipeline();
        System.out.println("Обнаружено " + outputs.size() + " котов");

        return rez;
    }

    public Mat matify(BufferedImage sourceImg) {

        java.awt.image.DataBuffer dataBuffer = sourceImg.getRaster().getDataBuffer();
        byte[] imgPixels = null;
        Mat imgMat = null;

        int width = sourceImg.getWidth();
        int height = sourceImg.getHeight();

        if (dataBuffer instanceof DataBufferByte) {
            imgPixels = ((DataBufferByte) dataBuffer).getData();
        }

        if (dataBuffer instanceof DataBufferInt) {

            int byteSize = width * height;
            imgPixels = new byte[byteSize * 3];

            int[] imgIntegerPixels = ((DataBufferInt) dataBuffer).getData();

            for (int p = 0; p < byteSize; p++) {
                imgPixels[p * 3 + 0] = (byte) ((imgIntegerPixels[p] & 0x00FF0000) >> 16);
                imgPixels[p * 3 + 1] = (byte) ((imgIntegerPixels[p] & 0x0000FF00) >> 8);
                imgPixels[p * 3 + 2] = (byte) (imgIntegerPixels[p] & 0x000000FF);
            }
        }

        if (imgPixels != null) {
            imgMat = new Mat(height, width, CvType.CV_8UC3);
            imgMat.put(0, 0, imgPixels);
        }

        return imgMat;
    }

    private static int argmax(List<Float> array) {
        float max = array.get(0);
        int re = 0;
        for (int i = 1; i < array.size(); i++) {
            if (array.get(i) > max) {
                max = array.get(i);
                re = i;
            }
        }
        return re;
    }

    private void setUnconnectedLayers() {

        for (Integer i : network.getUnconnectedOutLayers().toList()) {
            output_layers.add(layer_names.get(i - 1));
        }
    }

    private void setLayerNames() {
        layer_names = network.getLayerNames();
    }

    private void detectObject() {
        Mat blob_from_image = Dnn.blobFromImage(image, 0.00392, size, mean, true, false);
        network.setInput(blob_from_image);
        outputs = new ArrayList<Mat>();
        network.forward(outputs, output_layers);
        blob = blob_from_image;
    }

    private void getBoxDimensions() {
        for (Mat output : outputs) {
            System.out.println(output.height());
            for (int i = 0; i < output.height(); i++) {
                Mat row = output.row(i);
                MatOfFloat temp = new MatOfFloat(row);
                List<Float> detect = temp.toList();
                List<Float> score = detect.subList(5, 85);
                System.out.println( output.height());
                System.out.println(detect.size());
                System.out.println( score.size());
                int class_id = argmax(score);
                float conf = score.get(class_id);
                if (conf >= 0.4) {
                    int center_x = (int) (detect.get(0) * width);
                    int center_y = (int) (detect.get(1) * height);
                    int w = (int) (detect.get(2) * width);
                    int h = (int) (detect.get(3) * height);
                    int x = (center_x - w / 2);
                    int y = (center_y - h / 2);
                    Rect2d box = new Rect2d(x, y, w, h);
                    boxes.add(box);
                    confidences.add(conf);
                    class_ids.add(class_id);

                }

            }
        }
    }

    private void drawLabels() {
        double[] rgb = new double[]{255, 255, 0};
        Scalar color = new Scalar(rgb);
        MatOfRect2d mat = new MatOfRect2d();
        mat.fromList(boxes);
        MatOfFloat confidence = new MatOfFloat();
        confidence.fromList(confidences);
        MatOfInt indices = new MatOfInt();
        int font = Imgproc.FONT_HERSHEY_PLAIN;
        Dnn.NMSBoxes(mat, confidence, (float) (0.4), (float) (0.4), indices);
        List indices_list = indices.toList();
        System.out.println(indices_list.size());
        for (int i = 0; i < boxes.size(); i++) {
            if (indices_list.contains(i)) {
//                if (save) {
//                    Rect2d box = boxes.get(i);
//                    Point x_y = new Point(box.x, box.y);
//                    Point w_h = new Point(box.x + box.width, box.y + box.height);
//                    Point text_point = new Point(box.x, box.y - 5);
//                    Imgproc.rectangle(image, w_h, x_y, color);
//                    String label = classes.get(class_ids.get(i));
//                    Imgproc.putText(image, label, text_point, font, 1, color);
//                }

            }
        }
        if (save) {
            Imgcodecs.imwrite(output_path + "\\" + outputFileName + ".png.webp", image);
        }

    }

    public void loadPipeline() {
        try {
//            setNetwork();
            // загрузить классы?
            setClasses();
            setLayerNames();
            setUnconnectedLayers();
            loadImage();
            detectObject();
            getBoxDimensions();
            drawLabels();
        } catch (Exception e) {
            errors = true;
        }
    }

    private void setClasses() {
        try {
            File f = new File(class_file_name_dir);
            Scanner reader = new Scanner(f);
            while (reader.hasNextLine()) {
                String class_name = reader.nextLine();
                classes.add(class_name);
            }
        } catch (FileNotFoundException e) {
            errors = true;
        }
    }

    private void loadImage() {
        Mat mat = Imgcodecs.imread(input_path);
        Mat resizedImage = new Mat();
        Imgproc.resize(mat, resizedImage, size, 0.9, 0.9);
        height = resizedImage.height();
        width = resizedImage.width();
        channels = resizedImage.channels();
        image = resizedImage;
    }
}