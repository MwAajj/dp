package statistics;

import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.FileWriter;

public class Stats {
    private final static int random_size = 100;
    private final static String[] files = {
            "Covid_I"
    };
    private final static int k = 11;
    private static FileWriter writer;
    private final static String[][] myOptions = {
            {"-K", String.valueOf(k) },
            {"-K", String.valueOf(k), "-W"},
            {"-K", String.valueOf(k), "-F", "2"},
            {"-K", String.valueOf(k), "-D"},
            {"-K", String.valueOf(k), "-D", "-W"},
            {"-K", String.valueOf(k), "-D", "-F", "2"},
            {"-K", String.valueOf(k), "-B"},
            {"-K", String.valueOf(k), "-B", "-W"},
            {"-K", String.valueOf(k), "-B", "-F", "2"}
    };

    private final static String[][] wekaOption = {
            {"-K", String.valueOf(k)},
            {"-K", String.valueOf(k), "-I"},
            {"-K", String.valueOf(k), "-F"}
    };

    public static void main(String[] args) throws Exception {
        //test();

        for (int i = 0; i < files.length; i++) {
            String fileName = files[i];
            DatasetManager datasetManager = new DatasetManager(fileName, fileName, 0);

            InstanceManager manager = new InstanceManager(datasetManager.getOutputFileName(), 0);
            Instances train = manager.getTrain();
            Instances test = manager.getTest();
            Instances all = manager.getAll();

            double[][] sumMk = new double[random_size][myOptions.length];
            double[][] sumWeka = new double[random_size][wekaOption.length];


            writer = new FileWriter("src/main/resources/files/statistics/stat" + fileName + ".csv");
            for (int l = 0; l < random_size; l++) {
                EvaluationManager evaluation = new EvaluationManager(test, all, new String[]{"healthy", "ill"}, l);
                System.out.println(l);

                for (int j = 0; j < myOptions.length; j++) {
                    MyAlgorithm alg = new MyAlgorithm();
                    String[] option = myOptions[j];
                    alg.setOptions(option);
                    alg.buildClassifier(train);


                    System.out.println("-----------author---------");
                    evaluation.evaluateModel(alg);
                    evaluation.infoPrint();

                    double[] infoOption = evaluation.getInfoData();
                    if (l == 0)
                        sumMk[l][j] = infoOption[infoOption.length - 1];
                    else sumMk[l][j] = sumMk[l - 1][j] + infoOption[infoOption.length - 1];
                }

                for (int j = 0; j < wekaOption.length; j++) {
                    IBk weka = new IBk();
                    String[] option = wekaOption[j];
                    weka.setOptions(option);
                    weka.buildClassifier(train);
                    System.out.println("\n-----------WEKA---------");
                    evaluation.evaluateModel(weka);
                    evaluation.infoPrint();

                    double[] infoDataWeka = evaluation.getInfoData();
                    if (l == 0)
                        sumWeka[l][j] = infoDataWeka[infoDataWeka.length - 1];
                    else
                        sumWeka[l][j] = sumWeka[l - 1][j] + infoDataWeka[infoDataWeka.length - 1];
                }

                saveEvaluation(sumMk[l], sumWeka[l]);
            }
            writer.close();
        }
    }

    public static void saveEvaluation(double[] mk, double[] weka) throws Exception {
        for (int i = 0; i < mk.length; i++) {
            writer.append(String.valueOf(mk[i]));
            writer.append(";");
        }
        writer.append(";");
        for (int i = 0; i < weka.length; i++) {
            writer.append(String.valueOf(weka[i]));
            writer.append(";");
        }
        writer.append("\n");
    }
}
