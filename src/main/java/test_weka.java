import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Java program for classifying short text messages into two classes.
 */

public class test_weka implements Serializable {
    /** for serialization. */
    private static final long serialVersionUID = -123455813150452885L;
    /** The training data gathered so far. */
    private Instances m_Data = null;
    /** The filter used to generate the word counts. */
    private StringToWordVector m_Filter = new StringToWordVector();
    /** The actual classifier. */
    private Classifier m_Classifier = new J48();
    /** Whether the model is up to date. */
    private boolean m_UpToDate;
    /**
     * Constructs empty training dataset.
     */
    public test_weka() {
        String nameOfDataset = "MessageClassificationProblem";
// Create vector of attributes.
        ArrayList<Attribute> attributes = new ArrayList<Attribute>(2);
// Add attribute for holding messages.
        attributes.add(new Attribute("Message", (ArrayList<String>) null));
// Add class attribute.
        ArrayList<String> classValues = new ArrayList<String>(2);
        classValues.add("miss");
        classValues.add("hit");
        attributes.add(new Attribute("Class", classValues));
// Create dataset with initial capacity of 100, and set index of class.
        m_Data = new Instances(nameOfDataset, attributes, 100);
        m_Data.setClassIndex(m_Data.numAttributes() - 1);
    }

    /**
     * Updates model using the given training message.
     *
     * @param messagethe message content
     * @param classValuethe class label
     */
    public void updateData(String message, String classValue) {
// Make message into instance.
        Instance instance = makeInstance(message, m_Data);
// Set class value for instance.
        instance.setClassValue(classValue);
// Add instance to training data.
        m_Data.add(instance);
        m_UpToDate = false;
    }
    /**
     * Classifies a given message.
     *
     * @param messagethe message content
     * @throws Exception if classification fails
     */
    public void classifyMessage(String message) throws Exception {
// Check whether classifier has been built.
        if (m_Data.numInstances() == 0)
            throw new Exception("No classifier available.");
// Check whether classifier and filter are up to date.
        if (!m_UpToDate) {
// Initialize filter and tell it about the input format.
            m_Filter.setInputFormat(m_Data);
// Generate word counts from the training data.
            Instances filteredData = Filter.useFilter(m_Data, m_Filter);
// Rebuild classifier.
            m_Classifier.buildClassifier(filteredData);
            m_UpToDate = true;
        }
// Make separate little test set so that message
// does not get added to string attribute in m_Data.
        Instances testset = m_Data.stringFreeStructure();
// Make message into test instance.
        Instance instance = makeInstance(message, testset);
// Filter instance.
        m_Filter.input(instance);
        Instance filteredInstance = m_Filter.output();
// Get index of predicted class value.
        double predicted = m_Classifier.classifyInstance(filteredInstance);
// Output class value.
        System.err.println("Message classified as : " +
                m_Data.classAttribute().value((int) predicted));
    }

    /**
     * Method that converts a text message into an instance.
     *
     * @param textthe message content to convert
     * @param datathe header information
     * @returnthe generated Instance
     */
    private Instance makeInstance(String text, Instances data) {
// Create instance of length two.
        Instance instance = new DenseInstance(2);
// Set value for message attribute
        Attribute messageAtt = data.attribute("Message");
        instance.setValue(messageAtt, messageAtt.addStringValue(text));
// Give instance access to attribute information from the dataset.
        instance.setDataset(data);
        return instance;
    }
    /**
     * Main method. The following parameters are recognized:
     *
     * -m messagefile
     * Points to the file containing the message to classify or use for
     * updating the model.
     * -c classlabel
     * The class label of the message if model is to be updated. Omit for
     * classification of a message.
     * -t modelfile
     * The file containing the model. If it doesn't exist, it will be
     * created automatically.
     *
     * @param argsthe commandline options
     */
    public static void main(String[] args) {
        try {
// Read message file into string.
            String messageName = Utils.getOption('d', args);
//            if (messageName.length() == 0)
//                throw new Exception("Must provide name of data file ('-m <file>').");
//            FileReader m = new FileReader(messageName);
//            StringBuffer message = new StringBuffer();
//            int l;
//            while ((l = m.read()) != -1)
//                message.append((char) l);
//            m.close();
// Check if class value is given.
            String classValue = Utils.getOption('c', args);
// If model file exists, read it, otherwise create new one.
            String modelName = Utils.getOption('t', args);
            if (modelName.length() == 0)
                throw new Exception("Must provide name of model file ('-t <file>').");
            test_weka messageCl;
            try {
                messageCl = (test_weka) SerializationHelper.read(modelName);
            }
            catch (FileNotFoundException e) {
                messageCl = new test_weka();
            }
// Check if there are any options left
            Utils.checkForRemainingOptions(args);
// Process message.
//            if (classValue.length() != 0)
//                messageCl.updateData(message.toString(), classValue);
//            else
//                messageCl.classifyMessage(message.toString());
// Save message classifier object only if it was updated.
            if (classValue.length() != 0)
                SerializationHelper.write(modelName, messageCl);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}