package statistics;

import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Stats {
    private final static String[] files = {"Covid_I"};
    private final static int k = 3;
    private final static String[][] myOptions = {
            {"-K", String.valueOf(k)},
            /*{"-K", String.valueOf(k), "-W"},
            {"-K", String.valueOf(k), "-F", "2"},
            {"-K", String.valueOf(k), "-H"},*/
    };

    public static void main(String[] args) throws Exception {
        for (int i = 0; i < files.length; i++) {
            String fileName = files[i];
            DatasetManager datasetManager = new DatasetManager(fileName, fileName, 0);

            InstanceManager manager = new InstanceManager(datasetManager.getOutputFileName(), 0);
            Instances train = manager.getTrain();
            Instances test = manager.getTest();
            Instances all = manager.getAll();

            EvaluationManager evaluation = new EvaluationManager(test, all, new String[]{"healthy", "ill"});

            MyAlgorithm alg = new MyAlgorithm();
            IBk weka = new IBk(k);
            for (int j = 0; j < myOptions.length; j++) {
                alg.setOptions(myOptions[i]);
                alg.buildClassifier(train);
                System.out.println("-----------author---------");
                evaluation.evaluateModel(alg);
                evaluation.infoPrint();
                System.out.println("-----------WEKA---------");
                evaluation.evaluateModel(alg);
                evaluation.infoPrint();

                //evaluation.saveEvaluation();
            }
        }
    }
}
