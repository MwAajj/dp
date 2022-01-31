import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class WekaClassificationAlgorithm {

    public static void main(String[] args) throws Exception {
        String fileName = "src/main/resources/files/heart.arff";
        DataSource source = new DataSource(fileName);
        Instances instances = source.getDataSet();


        System.out.println(instances.toSummaryString());

        instances.setClassIndex(instances.numAttributes() - 1);

        //randomize order of records in my input dataset
        instances.randomize(new Random(23));

        //how much data in dataset, e.g. 80 percent
        int trainSize = (int) Math.round(instances.numInstances() * 0.80);
        int testSize = instances.numInstances() - trainSize;

        //from 0 to train size are train instances
        Instances trainInstances = new Instances(instances, 0, trainSize);

        //from train size to test size are test instances
        Instances testInstances = new Instances(instances, trainSize, testSize);


        Classifier classifier;

        //---------------------------KNN---------------------------
        classifier = new IBk();

        // ---------------------------NaiveBayes---------------------------
        classifier = new NaiveBayes();

        // ---------------------------Decision trees J48---------------------------
        classifier = new J48();

        classifier = new KNN();

        classifier.buildClassifier(trainInstances);


        Evaluation evaluation = new Evaluation(testInstances);
        evaluation.evaluateModel(classifier, testInstances);


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
