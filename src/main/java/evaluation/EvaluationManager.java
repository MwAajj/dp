package evaluation;

import lombok.Getter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.core.Instances;

import java.io.FileWriter;
import java.util.Random;

@Getter
public class EvaluationManager {

    private final int seed;


    enum Information {
        ACCURACY,
        PRECISION,
        F_MEASURE,
        RECALL,
        ESTIMATED_ACCURACY
    }

    private static final int INFO_SIZE = 5;

    private static final String[] infoName = {
            "Accuracy: ",
            "Precision: ",
            "F-measure: ",
            "Recall: ",
            "Estimated Accuracy: "
    };

    private final Instances test;
    private final Instances all;

    private double[] infoData;

    private final String[] results;
    private ConfusionMatrix confusionMatrix;

    public EvaluationManager(Instances test, Instances all, String[] results, int seed) {
        this.test = test;
        this.all = all;
        this.results = results;
        this.seed = seed;
    }


    public void evaluateModel(Classifier classifier) throws Exception {
        infoData = new double[INFO_SIZE];
        Evaluation evaluation = new Evaluation(test);
        evaluation.evaluateModel(classifier, test);


        confusionMatrix = new ConfusionMatrix(results);
        confusionMatrix.addPredictions(evaluation.predictions());


        infoData[Information.ACCURACY.ordinal()] = evaluation.pctCorrect();
        infoData[Information.PRECISION.ordinal()] = evaluation.precision(test.classIndex());
        infoData[Information.F_MEASURE.ordinal()] = evaluation.fMeasure(test.classIndex());
        infoData[Information.RECALL.ordinal()] = evaluation.recall(test.classIndex());

        evaluation.crossValidateModel(classifier, all, 10, new Random(seed));

        infoData[Information.ESTIMATED_ACCURACY.ordinal()] = evaluation.pctCorrect();
    }

    public void infoPrint() {
        for (int i = 0; i < infoData.length; i++) {
            System.out.println(infoName[i] + infoData[i]);
        }
    }

    public void infoPrintSoft() {
        System.out.println("\t\tEstimated Accuracy: " + infoData[Information.ESTIMATED_ACCURACY.ordinal()]);
    }
}
