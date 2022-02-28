package tests;

import structure.ballTree.BallTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

public class Test {
    private static Random rand;
    private static final int randomSize = 10_000;
    private static final int attrSize = 3;
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
            //if (i == 3435) {
            instanceArrayList = new Instances("Test", getAttr(), 2);
            baseInstances = new Instances("Test", getAttr(), 2);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            ballTree(i);
            // }
        }
    }

    private static void ballTree(int i) {
        BallTree ballTree = new BallTree(k);
        ballTree.buildTree(baseInstances);
        testNeighbours(ballTree, i);
    }

    private static void testNeighbours(BallTree ballTree, int i) {
        int j = 0;
        double z = 0;
        for (Instance instance : instanceArrayList) {
            //System.out.println("J: " + j);
            if (j == 0 && i == 3)
                z = 2;
            Instances kNearestNeighbours = ballTree.findKNearestNeighbours(instance, neighboursK);
            for (Instance kNearestNeighbour : kNearestNeighbours) {
                //System.out.println(kNearestNeighbour);
                for (int k= 0; k < kNearestNeighbours.numAttributes(); k++) {
                    double value = kNearestNeighbour.value(k);
                    if (value < BOTTOM_BORDER_K && neighboursK <= instancesSizeK)
                        System.out.println("ERROR");
                }
            }
            j++;
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
