package structure.kdtree;


import lombok.AllArgsConstructor;
import lombok.Getter;
import structure.MathOperation;
import structure.Tree;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.Queue;
import java.util.*;

public class KdTree extends NearestNeighbourSearch implements Tree {
    private boolean variance;
    private KdTreeNode root = null;
    private int classIndex = -1;
    int[] indices;
    double[] distances;
    private enum Son {
        LEFT,
        RIGHT
    }

    public KdTree(boolean variance) {
        this.variance = variance;
    }

    @Override
    public Instance nearestNeighbour(Instance target) {
        return findKNearestNeighbours(target, 1).firstInstance();
    }

    @Override
    public Instances kNearestNeighbours(Instance target, int k) {
        return findKNearestNeighbours(target, k);
    }

    @Override
    public double[] getDistances() {
        return this.distances;
    }

    /*Add instance to kdTree*/
    @Override
    public void update(Instance ins) throws Exception {
        if (root == null) {
            root.setInstance(ins);
            root.setLevel(indices[0]);
            return;
        }
        int level = indices[0];
        KdTreeNode node = root;
        while (true) {
            if(ins.value(level) <= node.getInstance().value(level)){
                if(node.getLeftSon() == null) {
                    node.setLeftSon(new KdTreeNode(ins));
                    node.getLeftSon().setParent(node);
                    node.getLeftSon().setLevel(getNewLevel(node));
                    break;
                }
                node = node.getLeftSon();
            } else {
                if(node.getRightSon() == null) {
                    node.setRightSon(new KdTreeNode(ins));
                    node.getRightSon().setParent(node);
                    node.getRightSon().setLevel(getNewLevel(node));
                }
                node = node.getRightSon();
            }
            level = getNewLevel(node);
        }
    }

    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }

    private void checkData(Instances data) {
        try {
            classIndex = data.classIndex();
        } catch (Exception e) {
            throw new RuntimeException("Instances doesn't have class index");
        }
        if(data.size() == 0)
            throw new RuntimeException("No data in instances");
    }

    @Override
    public void buildTree(Instances data) {
        checkData(data);

        if (!variance) indices = getIndices(data.firstInstance());
        else indices = getVariance(data);

        root = new KdTreeNode(data.firstInstance());
        Queue<KdTreeNode> nodeQueue = new LinkedList<>();
        KdTreeNode node;
        LinkedList<Instances> list = new LinkedList<>();
        list.add(data);

        Instance instance;

        if (classIndex == 0) {
            instance = MathOperation.getMedianInstance(new Instances(data), indices[1]);
            root.setLevel(indices[1]);
        } else {
            instance = MathOperation.getMedianInstance(new Instances(data), indices[1]);
            root.setLevel(indices[0]);
        }

        root.setInstance(instance);
        nodeQueue.add(root);

        for (int i = 0; i < list.size(); i++) {
            if (i > 0) {
                list.set(i - 1, null);
            }
            node = nodeQueue.poll();
            if (node == null) {
                System.out.println("Black Magic");
                return;
            }
            int level = node.getLevel();
            Instances[] arr = splitInstances(list.get(i), node.getInstance(), level);
            Instances leftInstances = arr[0];
            Instances rightInstances = arr[1];
            if (leftInstances.size() > 0)
                list.add(leftInstances);
            if (rightInstances.size() > 0)
                list.add(rightInstances);
            if (leftInstances.size() > 0) {
                level = getNewLevel(node);
                Instance leftInstance = MathOperation.getMedianInstance(leftInstances, level);
                node.setLeftSon(new KdTreeNode(leftInstance));
                node.getLeftSon().setLevel(level);
                node.getLeftSon().setParent(node);
                nodeQueue.add(node.getLeftSon());
            }

            if (rightInstances.size() > 0) {
                level = getNewLevel(node);
                Instance rightInstance = MathOperation.getMedianInstance(rightInstances, level);
                node.setRightSon(new KdTreeNode(rightInstance));
                node.getRightSon().setLevel(level);
                node.getRightSon().setParent(node);
                nodeQueue.add(node.getRightSon());
            }
        }
        System.out.println();
        //inOrderPrint();
    }

    private int[] getVariance(Instances data) {
        int[] result = new int[data.numAttributes()];
        double[] v = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                double variance = data.variance(i);
                v[i] = variance;
            } else {
                v[i] = Double.MIN_VALUE;
            }
        }
        Pair[] pairs = new Pair[data.numAttributes()];
        for (int i = 0; i < v.length; i++) {
            pairs[i] = new Pair(i, v[i]);
        }
        Arrays.sort(pairs);
        int j = 0;
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i == classIndex) {
                result[i] = classIndex;
            } else {
                if (pairs[j].getIndex() == classIndex) {
                    j++;
                }
                result[i] = pairs[j].getIndex();
                j++;
            }
        }
        return result;
    }

    @Getter
    @AllArgsConstructor
    private static class Pair implements Comparable<Pair> {
        private final int index;
        private final double value;

        @Override
        public int compareTo(Pair o) {
            return -1 * Double.compare(this.value, o.value);
        }
    }

    private int[] getIndices(Instance data) {
        ArrayList<Attribute> aLlAttributes = getALlAttributes(data);
        int[] arr = new int[aLlAttributes.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = i;
        }
        return arr;
    }

    @Override
    public Instances findKNearestNeighbours(Instance pInstance, int k) {
        Instances instances = initInstances(k);
        Stack<KdTreeNode> stack = new Stack<>();
        Stack<Son> visited = new Stack<>();
        KdTreeNode node = this.root;
        distances = new double[k];
        Arrays.fill(distances, Integer.MAX_VALUE);
        int level;
        double distance;

        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                level = node.getLevel();
                distance = MathOperation.euclidDistance(classIndex, node.getInstance(), pInstance);
                processDistance(instances, node.getInstance(), distance, distances);
                if (pInstance.value(level) <= node.getInstance().value(level)) {
                    if (node.getLeftSon() != null) {
                        node = node.getLeftSon();
                        visited.push(Son.LEFT);
                    } else {
                        node = null;
                        visited.push(Son.LEFT);
                    }
                } else {
                    if (node.getRightSon() != null) {
                        node = node.getRightSon();
                        visited.push(Son.RIGHT);
                    } else {
                        node = null;
                        visited.push(Son.RIGHT);
                    }
                }
            } else {
                node = stack.pop();
                Son visitedSon = visited.pop();
                if (node.isLeaf() // prevent from looping
                        || visitedSon == Son.LEFT && node.getRightSon() == null // prevent from crashing
                        || visitedSon == Son.RIGHT && node.getLeftSon() == null // prevent from crashing
                ) {
                    node = null; // prevent from looping
                    continue; // there is no hope for better point
                }

                distance = axisDistance(pInstance, node, visitedSon);
                if (distance < MathOperation.getMaxDistance(distances)) { // there is a hope to find a better node
                    node = visitedSon == Son.LEFT ? node.getRightSon() : node.getLeftSon(); //thanks to this I check every possible node 7,2,0
                } else {
                    node = null; // prevent from looping
                }
            }
        }
        //printNeighbours(instances, pInstance, distances);
        return instances;
    }


    private int getNewLevel(KdTreeNode node) {
        int level = node.getLevel();
        level++;
        if (level == classIndex) //don't organize tree by class index
            level++;
        level %= root.getInstance().numAttributes();
        level = indices[level];
        return level;
    }

    private Instances[] splitInstances(Instances nodeInstances, Instance medianInstance, int level) {
        Instances[] arrInstances = new Instances[2];
        int NODES_SIZE = 2;
        for (int i = 0; i < NODES_SIZE; i++) {
            arrInstances[i] = new Instances(String.valueOf(i), getALlAttributes(root.getInstance()), 2);
        }
        boolean setMedium = false;
        for (Instance instance : nodeInstances) {
            if (compareInstances(instance, medianInstance) && !setMedium) { // median instance must be excluded
                setMedium = true;
                continue;
            }

            if (instance.value(level) <= medianInstance.value(level)) {
                arrInstances[0].add(instance);
            } else {
                arrInstances[1].add(instance);
            }
        }
        return arrInstances;
    }

    private boolean compareInstances(Instance first, Instance second) {
        if (first.numAttributes() != second.numAttributes())
            throw new RuntimeException("Compare Instances: Incompatible size of instances");
        for (int i = 0; i < first.numAttributes(); i++) {
            if (first.value(i) != second.value(i))
                return false;
        }
        return true;
    }

    private Instances initInstances(int k) {
        Instances instances = new Instances("Neighbours", getALlAttributes(root.getInstance()), k);
        for (int i = 0; i < k; i++) {
            Instance initInstance = new DenseInstance(1, new double[root.getInstance().numAttributes()]);
            instances.add(initInstance);
        }
        instances.setClassIndex(root.getInstance().classIndex());
        return instances;
    }

    private void processDistance(Instances instances, Instance instance, double distance, double[] distances) {
        int index = -1;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < distances.length; i++) {
            if (max <= distances[i]) {
                max = distances[i];
                index = i;
            }
        }
        if (distance < max) { // better distance was founded
            distances[index] = distance;
            instances.set(index, instance);
        }
    }

    private double axisDistance(Instance pInstance, KdTreeNode node, Son visitedSon) {
        KdTreeNode son = visitedSon == Son.LEFT ? node.getRightSon() : node.getLeftSon();
        int level = node.getLevel();
        //for another axis 0 because we measure distance to section not to point
        return pInstance.value(level) - son.getInstance().value(level);
    }

    @Override
    public ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }

    private void printNeighbours(Instances instances, Instance pInstance, double[] distance) {
        System.out.println("Neighbours of instance: [" + pInstance + "]");
        int index = 0;
        for (Instance instance : instances) {
            System.out.print(instance);
            System.out.println(" \t with distance: " + distance[index]);
            index++;
        }
        System.out.println("--------------------------");
    }

    public void inOrderPrint() {
        System.out.println("--------------------IN ORDER-------------------------");
        Stack<KdTreeNode> stack = new Stack<>();
        KdTreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                node = node.getLeftSon();
            } else {
                node = stack.pop();
                System.out.println(node.getInstance());
                node = node.getRightSon();
            }
        }
        System.out.println("--------------------IN ORDER-------------------------");
    }
}
