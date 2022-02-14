package evaluation;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.core.Instances;

public class EvaluationManager {

    public EvaluationManager(Classifier classifier, Instances test) throws Exception {
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
