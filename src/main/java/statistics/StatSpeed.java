package statistics;

import classifier.EuclideanDistance;
import instance.InstanceManager;
import lombok.AllArgsConstructor;
import lombok.Getter;
import classifier.structure.Structure;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class StatSpeed {
    enum Variable {
        ATTR,
        K
    }

    enum Time {
        BRUTE_FORCE,
        KD_TREE,
        BALL_TREE,
        WEKA_BALL_TREE,
        WEKA_KD_TREE
    }

    private static final String[] text = {
            "Brute force sum: ",
            "KD structure sum: ",
            "Ball structure sum: ",
            "Weka Kd structure sum: ",
            "Weka ball structure sum: "
    };


    private static final int TIME_LENGTH = 5;
    private static Random rand;


    private static final int RANDOM_SIZE = 10;

    private static final boolean STAT_BUILD = false;
    private static final boolean IS_RANDOM = false;


    private static final int CLASS_INDEX = 0;

    private static int instancesSize = 10_000;
    private static final int INSTANCES_SIZE_K = 2_000;
    private static final DistanceFunction M_DISTANCE_FUNCTION = new EuclideanDistance();

    //private static final DistanceFunction M_DISTANCE_FUNCTION = new EuclideanDistance();

    private static final int ABOVE_BORDER = Integer.MAX_VALUE;
    private static final int BOTTOM_BORDER = 0;
    private static FileWriter resultFile;


    private static int attrSize;
    private static int neighboursK;
    private static String type;
    private static Instances baseInstances;

    private static long[][] times = new long[RANDOM_SIZE][TIME_LENGTH];
    private static long[] results = new long[TIME_LENGTH];

    private static final int[] neighbours = {
            3,
            10,
            20,
            50,
    };

    private static final int[] neighbours = {
            3,
            10,
            20,
            50,
    };

    private static final int[][] options = {
            {3, 3},
            {5, 5},
            {3, 10},
            {10, 3},
            {10, 10},
            {15, 15},
            {20, 3},
            {3, 20},
            {25, 25},
            {30, 30},
            {30, 5},
            {5, 30},
    };

    private static final String[] files =
            {
                    "Covid_I",
                    "Diabetes",
                    "Cardio",
                    "Gender",
                    "IRIS",
                    "Maternal",
            };

    private static String fileName;
    private static String sourceType;

    public static void main(String[] args) throws Exception {
        type = "_neighbour_";
        if (STAT_BUILD)
            type = "_build_";
        sourceType = "_random";

        if (IS_RANDOM)
            sourceType = "_dataset";
        resultFile = new FileWriter("src/main/resources/files/statistics/stat_results" + type + sourceType + ".csv");

        if (IS_RANDOM) {
            executeOptions();
            resultFile.close();
            return;
        }

        for (String file : files) {
            System.out.println("file: " + file);
            fileName = file;
            InstanceManager manager = new InstanceManager(fileName, CLASS_INDEX);
            baseInstances = manager.getTrain();
            executeFiles();
        }
        resultFile.close();
    }
    private static void executeFiles() throws  Exception{
        attrSize = baseInstances.numAttributes();
        for (int i = 0; i < neighbours.length; i++) {
            neighboursK = neighbours[i];
            execute();
        }
    }

    private static void executeOptions() throws Exception {
        for (int[] option : options) {
            times = new long[RANDOM_SIZE][TIME_LENGTH];
            results = new long[TIME_LENGTH];
            System.out.println("-------------------------------------------");
            attrSize = option[Variable.ATTR.ordinal()];
            neighboursK = option[Variable.K.ordinal()];
            execute();
        }
    }

    private static void execute() throws Exception {
        for (int i = 0; i < RANDOM_SIZE; i++) {
            System.out.println(i);
            rand = new Random(i);
            if (IS_RANDOM) {
                if (STAT_BUILD)
                    instancesSize = neighboursK;
                baseInstances = new Instances("Test", getAttr(), attrSize);
                setInstances();
            }
            bruteForce(i);
            kdTree(i);
            ballTree(i);
            ballTreeWeka(i);
            kdTreeWeka(i);
        }
        saveTimeValues();
    }

    private static void kdTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.KDTree structure = new weka.core.neighboursearch.KDTree();
        long l;
        if (STAT_BUILD)
            l = measureBuildWeka(structure);
        else {
            structure.setInstances(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.WEKA_KD_TREE.ordinal()] = l;
    }

    private static void ballTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.BallTree structure = new weka.core.neighboursearch.BallTree();
        long l;
        if (STAT_BUILD) {
            l = measureBuildWeka(structure);
        } else {
            structure.setInstances(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.WEKA_BALL_TREE.ordinal()] = l;
    }

    private static void bruteForce(int i) throws Exception {
        BruteForce structure = new BruteForce();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        long l;
        if (STAT_BUILD)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.BRUTE_FORCE.ordinal()] = l;
    }

    private static void kdTree(int i) throws Exception {
        KdTree structure = new KdTree(true);
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        long l;
        if (STAT_BUILD)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.KD_TREE.ordinal()] = l;
    }


    private static void ballTree(int i) throws Exception {
        BallTree structure = new BallTree();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        long l;
        if (STAT_BUILD)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.BALL_TREE.ordinal()] = l;
    }

    private static long measureBuildWeka(NearestNeighbourSearch structure) throws Exception {
        long start = System.currentTimeMillis();
        structure.setInstances(baseInstances);
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static long measureBuild(Structure structure) {
        long start = System.currentTimeMillis();
        structure.buildStructure(baseInstances);
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static long measureNeighbours(NearestNeighbourSearch structure) throws Exception {
        long start = System.currentTimeMillis();
        for (int a = 0; a < INSTANCES_SIZE_K; a++) {
            Instance instance = baseInstances.get(rand.nextInt(baseInstances.size()));
            structure.kNearestNeighbours(instance, neighboursK);
        }
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static void setInstances() {
        for (int m = 0; m < instancesSize; m++) {
            double[] values = new double[attrSize];
            for (int n = 0; n < attrSize; n++) {
                int val = BOTTOM_BORDER + rand.nextInt(ABOVE_BORDER - BOTTOM_BORDER);
                values[n] = val;
            }
            baseInstances.add(new DenseInstance(1d, values));
        }
        baseInstances.setClassIndex(CLASS_INDEX);
    }

    private static ArrayList<Attribute> getAttr() {
        ArrayList<Attribute> attr = new ArrayList<>(attrSize);
        for (int i = 0; i < attrSize; i++) {
            Attribute x = new Attribute(String.valueOf(i), i);
            attr.add(x);
        }
        return attr;
    }

    @AllArgsConstructor
    @Getter
    static class DistInst implements Comparable<DistInst>, Serializable {
        private Instance instance;
        private double distance;

        @Override
        public int compareTo(DistInst o) {
            return Double.compare(o.getDistance(), this.distance);
        }

    }


    private static void saveTimeValues() {
        FileWriter writer;
        long[][] sum = new long[RANDOM_SIZE][TIME_LENGTH];
        try {
            writer = new FileWriter("src/main/resources/files/statistics/stat" + type + sourceType + attrSize + "A___" + neighboursK + "K.csv");
            for (int i = 0; i < RANDOM_SIZE; i++) {
                for (int j = 0; j < times[i].length; j++) {
                    if (i == 0) sum[i][j] += times[i][j];
                    else sum[i][j] = sum[i - 1][j] + times[i][j];
                    writer.append(String.valueOf(times[i][j]));
                    writer.append(";");

                }
                for (int j = 0; j < times[i].length; j++) {
                    writer.append(String.valueOf(sum[i][j]));
                    writer.append(";");
                }
                writer.append("\n");
            }
            writer.close();

            for (int i = 0; i < results.length; i++) {
                results[i] = times[RANDOM_SIZE - 1][i];

                resultFile.append(String.valueOf(results[i]));
                resultFile.append(";");
            }
            resultFile.append("\n");
        } catch (Exception e) {
            System.out.println("Exception " + e);
        }
        for (int i = 0; i < TIME_LENGTH; i++) {
            System.out.println(text[i] + sum[RANDOM_SIZE - 1][i]);
        }
    }
}
