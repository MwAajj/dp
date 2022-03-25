package tests;

import classifier.EuclideanDistance;
import classifier.structure.Structure;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.core.*;

import java.util.ArrayList;
import java.util.Random;

public class StructureTest {
    private static Random rand;
    private static final int RANDOM_SIZE = 100;
    private static final int ATTR_SIZE = 30;

    private static final int PVD = 99;
    private static final int NEIGHBOURS_K = 111;
    private static final int CLASS_INDEX = 0;

    private static final int INSTANCES_SIZE_K = 111;

    public static final int ABOVE_BORDER = 10;
    public static final int BOTTOM_BORDER = 0;

    public static final int ABOVE_BORDER_K = 100;
    public static final int BOTTOM_BORDER_K = 90;

    public static final int ABOVE_RANDOM = 100;
    public static final int BOTTOM_RANDOM = 0;

    private static Instances instanceArrayList;
    private static Instances baseInstances;

    private static final DistanceFunction M_DISTANCE_FUNCTION = new EuclideanDistance();

    //private static final DistanceFunction M_DISTANCE_FUNCTION = new EuclideanDistance();


    public static void main(String[] args) {
        for (int i = 0; i < RANDOM_SIZE; i++) {
            instanceArrayList = new Instances("Test", getAttr(), 2);
            baseInstances = new Instances("Test", getAttr(), 2);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            bruteForce();
            ballTree();
            kdTree();
        }
    }

    private static void bruteForce() {
        Structure structure = new BruteForce();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        testNeighbours(structure);
    }

    private static void kdTree() {
        KdTree structure = new KdTree(false);
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        testNeighbours(structure);
    }


    private static void ballTree() {
        BallTree structure = new BallTree();
        structure.setDistanceFunction(M_DISTANCE_FUNCTION);
        structure.buildStructure(baseInstances);
        testNeighbours(structure);
    }

    private static void testNeighbours(Structure structure) {
        for (Instance instance : instanceArrayList) {
            Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, NEIGHBOURS_K);
            if (kNearestNeighbours.size() != NEIGHBOURS_K)
                throw new RuntimeException("Error in size");
            for (Instance kNearestNeighbour : kNearestNeighbours) {
                for (int k = 0; k < kNearestNeighbours.numAttributes(); k++) {
                    double value = kNearestNeighbour.value(k);
                    if (value < BOTTOM_BORDER_K)
                        throw new RuntimeException("Error in data");
                }
            }
        }
    }

    private static void setInstances() {
        while (instanceArrayList.size() < INSTANCES_SIZE_K) {
            double[] values = new double[ATTR_SIZE];
            int ran = rand.nextInt(((ABOVE_RANDOM - BOTTOM_RANDOM) + 1)) + BOTTOM_RANDOM;
            if (ran > PVD && instanceArrayList.size() < INSTANCES_SIZE_K) {
                for (int j = 0; j < ATTR_SIZE; j++) {
                    int val = rand.nextInt(((ABOVE_BORDER_K - BOTTOM_BORDER_K) + 1)) + BOTTOM_BORDER_K;
                    values[j] = val;
                }
                baseInstances.add(new DenseInstance(1d, values));
                instanceArrayList.add(new DenseInstance(1d, values));
            } else {
                for (int j = 0; j < ATTR_SIZE; j++) {
                    int val = rand.nextInt(((ABOVE_BORDER - BOTTOM_BORDER) + 1)) + BOTTOM_BORDER;
                    values[j] = val;
                }
                baseInstances.add(new DenseInstance(1d, values));
            }
        }
        baseInstances.setClassIndex(CLASS_INDEX);
        instanceArrayList.setClassIndex(CLASS_INDEX);
    }

    private static ArrayList<Attribute> getAttr() {
        ArrayList<Attribute> attr = new ArrayList<>(ATTR_SIZE);
        for (int i = 0; i < ATTR_SIZE; i++) {
            Attribute x = new Attribute(String.valueOf(i), i);
            attr.add(x);
        }
        return attr;
    }


}
