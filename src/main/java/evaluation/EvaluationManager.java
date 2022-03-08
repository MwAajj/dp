package evaluation;

import classifier.MyAlgorithm;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.core.Instances;

import java.util.Random;

public class EvaluationManager {

    enum Information {
        ACCURACY,
        PRECISION,
        F_MEASURE,
        RECALL,
        ESTIMATED_ACCURACY
    }

    private static final int INFO_SIZE = 4;

    private static final String[] infoName = {
            "Accuracy: ",
            "Precision: ",
            "F-measure: ",
            "Recall: ",
            "Estimated Accuracy: "
    };

    private Instances test;
    private Instances all;

    private double[] infoData = new double[4];

    private String[] results;
    private ConfusionMatrix confusionMatrix;


    public void evaluateModel(Classifier classifier) throws Exception {

        Evaluation evaluation = new Evaluation(test);
        evaluation.evaluateModel(classifier, test);
        System.out.println(evaluation.toSummaryString());

        confusionMatrix = new ConfusionMatrix(results);
        confusionMatrix.addPredictions(evaluation.predictions());


        infoData[Information.ACCURACY.ordinal()] = evaluation.pctCorrect();
        infoData[Information.ACCURACY.ordinal()] = evaluation.precision(1);
        infoData[Information.ACCURACY.ordinal()] = evaluation.fMeasure(1);
        infoData[Information.ACCURACY.ordinal()] = evaluation.recall(1);

        evaluation.crossValidateModel(classifier, all, 10, new Random(0));

        infoData[Information.ACCURACY.ordinal()] = evaluation.pctCorrect();
    }

    public void infoPrint() {
        System.out.println(confusionMatrix);
        System.out.println();
        for (int i = 0; i < infoData.length; i++) {
            System.out.println(infoName[i] + infoData[i]);
        }
    }



    public EvaluationManager(Instances test, Instances all, String[] results) {
        this.test = test;
        this.all = all;
        this.results = results;
    }

}
