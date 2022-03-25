import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;

public class Source {
    public static void main(String[] args) throws Exception {
        System.out.println("KNN Extension");
        System.out.println("\t----------My Implementation-----------------");

        String fileName = "Covid_I";
        DatasetManager dm = new DatasetManager(fileName, 0);

        InstanceManager iM = new InstanceManager(dm.getOutputFileName(), 0);


        Classifier classifier = new MyAlgorithm(9);
        test(classifier, iM);

        System.out.println("\t----------WEKA-----------------");
        classifier = new IBk(9);
        test(classifier, iM);
    }

    private static void test(Classifier classifier, InstanceManager iM) throws Exception {
        classifier.buildClassifier(iM.getTrain());
        EvaluationManager eM =
                new EvaluationManager(iM.getTest(), iM.getAll(), new String[]{"healthy", "ill"}, 10);
        eM.evaluateModel(classifier);
        eM.infoPrintSoft();
    }
}
