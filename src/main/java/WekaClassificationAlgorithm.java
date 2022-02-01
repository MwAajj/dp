import dataset.ArffManager;
import dataset.DatasetManager;
import instance.InstanceManager;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class WekaClassificationAlgorithm {

    public static void main(String[] args) throws Exception {
        //kriterium - shannova entropia, klassification error, giny index, fuzzy rozhodovacia strom,
        Classifier classifier;

        //---------------------------KNN---------------------------
        //classifier = new IBk();

        // ---------------------------NaiveBayes---------------------------
        //classifier = new NaiveBayes();

        // ---------------------------Decision trees J48---------------------------
        classifier = new J48();

        //classifier = new MyAlgorithm();
        DatasetManager dataset = new DatasetManager(true);
        String fileName = "testData";

        InstanceManager manager = new InstanceManager(fileName);

        Instances all = manager.getAll();
        Instances test = manager.getTest();
        Instances train = manager.getTrain();

        //manager.getNewInstance();

        classifier.buildClassifier(train);


        //epoch for bigger data
        //batch - zhluk dat
        //filtracia dat
        //normalizacia vstupnych dat preco sa to robi
        //aby mal lepsiu formu ucenia algoritmus

        Evaluation evaluation = new Evaluation(test);
        evaluation.evaluateModel(classifier, test);
        //toto si viem ulozit
        // dovod ulozenia matice




        System.out.println(evaluation.toSummaryString());

        //Validation via confusion matrix

        ConfusionMatrix confusionMatrix = new ConfusionMatrix(new String[] {"Healthy", "Ill"});
        confusionMatrix.addPredictions(evaluation.predictions());

        System.out.println(confusionMatrix);
        System.out.println("Accuracy: " + evaluation.pctCorrect());
        System.out.println("Precision: " + evaluation.precision(1));
        System.out.println("Recall: " + evaluation.recall(1));
    }
}
