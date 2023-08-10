import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgcodecs.Imgcodecs;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        CascadeClassifier face_cascade = new CascadeClassifier("haarcascade_frontalcatface.xml");
        if(face_cascade.empty()){
            System.out.println("Ошибка загрузки");
            return;
        }
        else{
            System.out.println("Загрузили \"haarcascade_frontalcatface\"");
        }

        Mat inputFrame = Imgcodecs.imread(args[0]);
        Mat mRgba=new Mat();
        Mat mGrey=new Mat();
        MatOfRect faces = new MatOfRect();

        inputFrame.copyTo(mRgba);
        inputFrame.copyTo(mGrey);

        Imgproc.cvtColor( mRgba, mGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist( mGrey, mGrey );

        face_cascade.detectMultiScale(mGrey, faces);
        int detectedFaces = faces.toArray().length;

        System.out.println("Обнаружено " + detectedFaces + " котов");

        face_cascade = new CascadeClassifier("haarcascade_frontalcatface_extended.xml");
        if(face_cascade.empty()){
            System.out.println("Ошибка загрузки");
            return;
        }
        else{
            System.out.println("Загрузили \"haarcascade_frontalcatface_extended\"");
        }

        face_cascade.detectMultiScale(mGrey, faces);
        detectedFaces = faces.toArray().length;

        System.out.println("Обнаружено " + detectedFaces + " котов");
        System.exit(0);
    }
}