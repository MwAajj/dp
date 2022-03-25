package tests;

import lombok.AllArgsConstructor;
import lombok.Getter;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class CompareTest {
    private static Random rand;
    private static final int RANDOM_SIZE = 1_000;

    private static final int ATTR_SIZE = 5;
    private static final int NEIGHBOURS_K = 11;
    private static final int CLASS_INDEX = 0;

    private static final int INSTANCES_SIZE = 1_000;
    private static final int INSTANCES_SIZE_K = 2_00;

    public static final int ABOVE_BORDER = Integer.MAX_VALUE;
    public static final int BOTTOM_BORDER = 0;

    private static Instances baseInstances;
    private static DistInst[][] bruteForceInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
    private static DistInst[][] kdInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
    private static DistInst[][] ballInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
    private static Instances[] wekaInstances = new Instances[INSTANCES_SIZE_K];
    private static final double THRESHOLD = 0.000001d;


    private static final DistanceFunction M_DISTANCE_FUNCTION = new classifier.EuclideanDistance();

    //private static final DistanceFunction M_DISTANCE_FUNCTION = new EuclideanDistance();


    public static void main(String[] args) throws Exception {
        for (int i = 0; i < RANDOM_SIZE; i++) {
            bruteForceInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
            kdInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
            ballInstances = new DistInst[INSTANCES_SIZE_K][NEIGHBOURS_K];
            baseInstances = new Instances("Test", getAttr(), ATTR_SIZE);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            bruteForce();
            kdTree();
            ballTree();
            wekaKd();
            sort();
            compare();
        }
    }

    private static void wekaKd() throws Exception {
        for (int a = 0; a < INSTANCES_SIZE_K; a++) {
            Instance instance = baseInstances.get(a);
            KDTree kdTree = new KDTree();
            kdTree.setInstances(baseInstances);
            Instances instances = kdTree.kNearestNeighbours(instance, NEIGHBOURS_K);
            wekaInstances[a] = instances;
        }
    }

    private static void sort() {
        for (int i = 0; i < INSTANCES_SIZE_K; i++) {
            Arrays.sort(bruteForceInstances[i], Collections.reverseOrder());
            Arrays.sort(kdInstances[i], Collections.reverseOrder());
            Arrays.sort(ballInstances[i], Collections.reverseOrder());
        }
    }

    private static void compare() {
        for (int i = 0; i < INSTANCES_SIZE_K; i++) {
            for (int j = 0; j < NEIGHBOURS_K; j++) {
                //threshold because of comparing doubles for test purposes
                if (Math.abs(bruteForceInstances[i][j].getDistance() - kdInstances[i][j].getDistance()) > THRESHOLD) {
                    throw new RuntimeException("Kd mistake");
                }
                if (Math.abs(bruteForceInstances[i][j].getDistance() - ballInstances[i][j].getDistance()) > THRESHOLD) {
                    throw new RuntimeException("Ball Mistake");
                }
            }
        }
    }

    private static void test(NearestNeighbourSearch structure, int x) throws Exception {
        for (int a = 0; a < INSTANCES_SIZE_K; a++) {
            Instance instance = baseInstances.get(a);
            Instances instances = structure.kNearestNeighbours(instance, NEIGHBOURS_K);
            double[] distances = structure.getDistances();
            addInstances(x, a, instances, distances);
        }
    }

    private static void addInstances(int x, int a, Instances instances, double[] distances) {
        for (int i = 0; i < NEIGHBOURS_K; i++) {
            if (x == 0)
                bruteForceInstances[a][i] = new DistInst(instances.get(i), distances[i]);
            if (x == 1)
                kdInstances[a][i] = new DistInst(instances.get(i), distances[i]);
            if (x == 2)
                ballInstances[a][i] = new DistInst(instances.get(i), distances[i]);
        }
    }

    private static void bruteForce() throws Exception {
        BruteForce structure = new BruteForce();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        test(structure, 0);
    }

    private static void kdTree() throws Exception {
        KdTree structure = new KdTree(false);
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        test(structure, 1);
    }


    private static void ballTree() throws Exception {
        BallTree structure = new BallTree();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        test(structure, 2);
    }


    private static void setInstances() {
        for (int m = 0; m < INSTANCES_SIZE; m++) {
            double[] values = new double[ATTR_SIZE];
            for (int n = 0; n < ATTR_SIZE; n++) {
                int val = BOTTOM_BORDER + rand.nextInt(ABOVE_BORDER - BOTTOM_BORDER);
                values[n] = val;
            }
            baseInstances.add(new DenseInstance(1d, values));
        }
        baseInstances.setClassIndex(CLASS_INDEX);
    }

    private static ArrayList<Attribute> getAttr() {
        ArrayList<Attribute> attr = new ArrayList<>(ATTR_SIZE);
        for (int i = 0; i < ATTR_SIZE; i++) {
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
            return Double.compare(o.distance, this.distance);
        }
    }
}
