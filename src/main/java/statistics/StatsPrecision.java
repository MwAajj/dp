package statistics;

import classifier.MyAlgorithm;
import classifier.structure.trees.ballTree.BallTreeNode;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.BallTree;

import java.io.FileWriter;

public class StatsPrecision {
    private final static int random_size = 10;
    private final static String[] files = {
            //"IRIS"
            //"EMG"
            //"Covid_I"
            //"Maternal"
            //"EEG"
            //"Diabetes"
            "heart"

    };
    private final static String[][] results = {
            //{"Setosa", "Versicolor", "Virginica"}
            //{"normal", "suspect", "pathologic"}
            //{"healthy", "ill"}
            //{"high risk", "low risk", "mid risk"}
            //{"healthy", "ill"}
            //{"healthy", "ill"}
            //{"dead", "alive"}
            {"healthy", "ill"}
    };
    private final static int[] kVariables = {5};
    private static FileWriter writer;

    public static void main(String[] args) throws Exception {
        //test();

        for (int i = 0; i < files.length; i++) {
            String fileName = files[i];
            DatasetManager datasetManager = new DatasetManager(fileName, fileName, 0);

            InstanceManager manager = new InstanceManager(datasetManager.getOutputFileName(), 0);
            Instances train = manager.getTrain();
            Instances test = manager.getTest();
            Instances all = manager.getAll();
            for (int k : kVariables) {
                String newFilename;
                newFilename = fileName + "_" + k;
                String[][] myOptions = {
                        /*{"-K", String.valueOf(k)},
                        {"-K", String.valueOf(k), "-W"},
                        {"-K", String.valueOf(k), "-F", "2"},
                        {"-K", String.valueOf(k), "-H"},*/
                        /*{"-K", String.valueOf(k), "-D"},
                        {"-K", String.valueOf(k), "-D", "-W"},
                        {"-K", String.valueOf(k), "-D", "-F", "2"},
                        {"-K", String.valueOf(k), "-D", "-H"},*/
                        {"-K", String.valueOf(k), "-B"},
                        /*{"-K", String.valueOf(k), "-B", "-W"},
                        {"-K", String.valueOf(k), "-B", "-F", "2"},
                        {"-K", String.valueOf(k), "-B", "-H"},*/
                };

                String[][] wekaOption = {
                        {"-K", String.valueOf(k)}
                        /*{"-K", String.valueOf(k), "-I"},
                        {"-K", String.valueOf(k), "-F"}*/
                };


                double[][] sumMk = new double[random_size][myOptions.length];
                double[][] sumWeka = new double[random_size][wekaOption.length];


                writer = new FileWriter("src/main/resources/files/statistics/stat" + newFilename + ".csv");
                for (int l = 0; l < random_size; l++) {
                    EvaluationManager evaluation = new EvaluationManager(test, all, results[i], l);
                    System.out.println("L: " + l);

                    for (int j = 0; j < myOptions.length; j++) {
                        System.out.println("\t M J:" + j);
                        MyAlgorithm alg = new MyAlgorithm();
                        String[] option = myOptions[j];
                        alg.setOptions(option);
                        alg.buildClassifier(train);


                        //System.out.println("-----------author---------");
                        evaluation.evaluateModel(alg);
                        //evaluation.infoPrint();

                        double[] infoOption = evaluation.getInfoData();
                        if (l == 0)
                            sumMk[l][j] = infoOption[infoOption.length - 1];
                        else sumMk[l][j] = sumMk[l - 1][j] + infoOption[infoOption.length - 1];
                    }

                    for (int j = 0; j < wekaOption.length; j++) {
                        System.out.println("\t W J:" + j);
                        IBk weka = new IBk();
                        weka.setNearestNeighbourSearchAlgorithm(new BallTree());
                        weka.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new EuclideanDistance());
                        String[] option = wekaOption[j];
                        weka.setOptions(option);
                        weka.buildClassifier(train);
                        //System.out.println("\n-----------WEKA---------");
                        evaluation.evaluateModel(weka);
                        //evaluation.infoPrint();

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
    }

    public static void saveEvaluation(double[] mk, double[] weka) throws Exception {
        for (double v : mk) {
            System.out.println("MK: " + v);
            writer.append(String.valueOf(v));
            writer.append(";");
        }
        writer.append(";");
        for (double v : weka) {
            System.out.println("WEKA: " + v);
            writer.append(String.valueOf(v));
            writer.append(";");
        }
        writer.append("\n");
    }
}
