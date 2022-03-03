package tests;

import structure.MathOperation;
import structure.Tree;
import structure.ballTree.BallTree;
import structure.kdtree.KdTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class StructureTest {
    private static Random rand;
    private static final int randomSize = 1000;
    private static final int attrSize = 30;
    private static final int k = 3;
    private static final int neighboursK = 111;
    private static final int classIndex = 0;

    private static final int instancesSizeK = 111;

    public static final int ABOVE_BORDER = 10;
    public static final int BOTTOM_BORDER = 0;

    public static final int ABOVE_BORDER_K = 100;
    public static final int BOTTOM_BORDER_K = 90;

    public static final int ABOVE_RANDOM = 100;
    public static final int BOTTOM_RANDOM = 0;

    private static Instances instanceArrayList;
    private static Instances baseInstances;

    public static void main(String[] args) {
        for (int i = 0; i < randomSize; i++) {
            instanceArrayList = new Instances("Test", getAttr(), 2);
            baseInstances = new Instances("Test", getAttr(), 2);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            bruteForceNeighbour();
            //ballTree(i);
            //kdTree(i);
        }
    }

    private static void bruteForceNeighbour() {
        Instances kNearestNeighbours = new Instances("Test", getAttr(), baseInstances.size());
        double[] neighbours = new double[baseInstances.size()];
        Arrays.fill(neighbours, Double.MAX_VALUE);
        for (int i = 0; i < neighboursK; i++) {
            kNearestNeighbours.add(new DenseInstance(1d, neighbours));
        }
        for (int j = 0; j < instancesSizeK; j++) {
            Instance target = baseInstances.get(rand.nextInt(baseInstances.size()));
            for (Instance instance : baseInstances) {
                double v = MathOperation.euclidDistance(classIndex, instance, target);
                neighbours[j] = v;
                kNearestNeighbours.set(j, instance);
            }
        }
        Arrays.sort(neighbours);
        for (Instance kNearestNeighbour : kNearestNeighbours) {
            for (int k = 0; k < kNearestNeighbours.numAttributes(); k++) {
                double value = kNearestNeighbour.value(k);
                if (value < BOTTOM_BORDER_K)
                    throw new RuntimeException("Error in data");
            }
        }
    }

    private static void kdTree(int i) {
        KdTree kdTree = new KdTree(false);
        kdTree.buildTree(baseInstances);
        testNeighbours(kdTree, i);
    }


    private static void ballTree(int i) {
        BallTree ballTree = new BallTree(k);
        ballTree.buildTree(baseInstances);
        testNeighbours(ballTree, i);
    }

    private static void testNeighbours(Tree tree, int i) {
        //int j  = 0;
        for (Instance instance : instanceArrayList) {
            //System.out.println("\t J: " + j);
            Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, neighboursK);
            if (kNearestNeighbours.size() != neighboursK)
                throw new RuntimeException("Error in size");
            for (Instance kNearestNeighbour : kNearestNeighbours) {
                for (int k = 0; k < kNearestNeighbours.numAttributes(); k++) {
                    double value = kNearestNeighbour.value(k);
                    if (value < BOTTOM_BORDER_K)
                        throw new RuntimeException("Error in data");
                }
            }
            //j++;
        }
    }

    private static void setInstances() {
        while (instanceArrayList.size() < instancesSizeK) {
            double[] values = new double[attrSize];
            double ran = rand.nextInt(((ABOVE_RANDOM - BOTTOM_RANDOM) + 1)) + BOTTOM_RANDOM;
            if (ran > 99 && instanceArrayList.size() < instancesSizeK) {
                for (int j = 0; j < attrSize; j++) {
                    double val = rand.nextInt(((ABOVE_BORDER_K - BOTTOM_BORDER_K) + 1)) + BOTTOM_BORDER_K;
                    values[j] = val;
                }
                baseInstances.add(new DenseInstance(1d, values));
                instanceArrayList.add(new DenseInstance(1d, values));
            } else {
                for (int j = 0; j < attrSize; j++) {
                    double val = rand.nextInt(((ABOVE_BORDER - BOTTOM_BORDER) + 1)) + BOTTOM_BORDER;
                    values[j] = val;
                }
                baseInstances.add(new DenseInstance(1d, values));
            }
        }
        baseInstances.setClassIndex(classIndex);
        instanceArrayList.setClassIndex(classIndex);
    }

    private static ArrayList<Attribute> getAttr() {
        ArrayList<Attribute> attr = new ArrayList<>(attrSize);
        for (int i = 0; i < attrSize; i++) {
            Attribute x = new Attribute(String.valueOf(i), i);
            attr.add(x);
        }
        return attr;
    }


}