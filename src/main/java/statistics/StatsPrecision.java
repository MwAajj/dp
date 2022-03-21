package statistics;

import classifier.MyAlgorithm;
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
            "Covid_I",
            "Diabetes",
            "Cardio",
            "Gender",
            "IRIS",
            "Maternal"
    };
    private final static String[][] classOptions = {
            {"healthy", "ill"},
            {"healthy", "ill"},
            {"normal", "suspect", "pathologic"},
            {"male", "female"},
            {"Setosa", "Versicolor", "Virginica"},
            {"high risk", "low risk", "mid risk"}
    };

    private final static int[] kVariables = {3, 7, 11};

    public static void main(String[] args) throws Exception {
        //test();
        FileWriter resultFile = new FileWriter("src/main/resources/files/statistics/statResultPrecision.csv");
        for (int i = 0; i < files.length; i++) {
            String fileName = files[i];
            System.out.println("Files" + fileName);
            DatasetManager datasetManager = new DatasetManager(fileName, fileName, 0);

            InstanceManager manager = new InstanceManager(datasetManager.getOutputFileName(), 0);
            Instances train = manager.getTrain();
            Instances test = manager.getTest();
            Instances all = manager.getAll();
            for (int k : kVariables) {
                String newFilename;
                newFilename = fileName + "_" + k;
                String[][] myOptions = {
                        {"-K", String.valueOf(k), "-B"},
                        {"-K", String.valueOf(k), "-B", "-W"},
                        {"-K", String.valueOf(k), "-B", "-F", "2"},
                        {"-K", String.valueOf(k), "-B", "-H"},
                };

                String[][] wekaOption = {
                        {"-K", String.valueOf(k)},
                        {"-K", String.valueOf(k), "-I"},
                        {"-K", String.valueOf(k), "-F"}
                };


                double[][] sumMk = new double[random_size][myOptions.length];
                double[][] sumWeka = new double[random_size][wekaOption.length];


                FileWriter writer = new FileWriter("src/main/resources/files/statistics/stat" + newFilename + ".csv");
                for (int l = 0; l < random_size; l++) {
                    EvaluationManager evaluation = new EvaluationManager(test, all, classOptions[i], l);
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
                    saveEvaluation(writer, sumMk[l], sumWeka[l]);
                }
                saveEvaluation(resultFile, sumMk[random_size - 1], sumWeka[random_size - 1]);
                writer.close();
            }
        }
        resultFile.close();
    }

    public static void saveEvaluation(FileWriter writer, double[] mk, double[] weka) throws Exception {
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
