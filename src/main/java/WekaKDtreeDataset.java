import dataset.DatasetManager;
import instance.InstanceManager;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

public class WekaKDtreeDataset {

    public static void main(String[] args) throws Exception {

        String fileName = "testData2";

        InstanceManager manager = new InstanceManager(fileName);
        Instances train = manager.getTrain();
        Instances test = manager.getTest();
        Instances all = manager.getAll();
        manager.printInstances();

        KDTree knn = new KDTree();
        knn.setInstances(train);

        MyAlgorithm classifier = new MyAlgorithm();
        String[] options = new String[1];
        options[0] ="-K";
        classifier.setOptions(options);
        classifier.buildClassifier(all);

        Evaluation evaluation = new Evaluation(test);

        evaluation.evaluateModel(classifier, test);
        System.out.println(evaluation.toSummaryString());



        ConfusionMatrix confusionMatrix = new ConfusionMatrix(new String[] {"Healthy", "Ill"});
        confusionMatrix.addPredictions(evaluation.predictions());

        System.out.println(confusionMatrix);
        System.out.println("Accuracy: " + evaluation.pctCorrect());
        System.out.println("Precision: " + evaluation.precision(1));
        System.out.println("Recall: " + evaluation.recall(1));

    }
}
