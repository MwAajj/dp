package structure.kdtree;

import structure.MathOperation;
import structure.Tree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;

public class KdTree implements Tree, Serializable{

    private KdTreeNode root = null;
    private int NODES_SIZE = 2;
    private int classIndex = -1;

    private enum Son {
        LEFT,
        RIGHT
    }

    //random atribut na zaklade rozptylu to rob
    @Override
    public void buildTree(Instances data) {
        root = new KdTreeNode(data.firstInstance());
        try {
            classIndex = data.classIndex();
        } catch (Exception e) {
            throw new RuntimeException("Instances doesn't have class index");
        }
        Queue<KdTreeNode> nodeQueue = new LinkedList<>();

        KdTreeNode node;
        LinkedList<Instances> list = new LinkedList<>();
        list.add(data);
        Instance instance = MathOperation.getMedianInstance(new Instances(data), 0);

        root.setInstance(instance);
        if(classIndex == 0)
            root.setLevel(1);
        else root.setLevel(0);
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
                level = getLevel(node);
                Instance leftInstance = MathOperation.getMedianInstance(leftInstances, level);
                node.setLeftSon(new KdTreeNode(leftInstance));
                node.getLeftSon().setLevel(level);
                node.getLeftSon().setParent(node);
                nodeQueue.add(node.getLeftSon());
            }

            if (rightInstances.size() > 0) {
                level = getLevel(node);
                Instance rightInstance = MathOperation.getMedianInstance(rightInstances, level);
                node.setRightSon(new KdTreeNode(rightInstance));
                node.getRightSon().setLevel(level);
                node.getRightSon().setParent(node);
                nodeQueue.add(node.getRightSon());
            }
        }
        int x = -1;
        //System.out.println("End");
        //inOrderPrint();
    }

    @Override
    public Instances findKNearestNeighbours(Instance pInstance, int k) {
        Instances instances = initInstances(k);
        Stack<KdTreeNode> stack = new Stack<>();
        Stack<Son> visited = new Stack<>();
        KdTreeNode node = this.root;
        double[] distances = new double[k];
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


    private int getLevel(KdTreeNode node) {
        int level = node.getLevel();
        level++;
        if(level == classIndex) //don't organize tree by class index
            level++;
        level %= root.getInstance().numAttributes();
        return level;
    }

    private Instances[] splitInstances(Instances nodeInstances, Instance medianInstance, int level) {
        Instances[] arrInstances = new Instances[2];
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
            Instance initInstance = new DenseInstance(1, new double[root.getInstance().numAttributes()]); //TODO  WEIGHT 1 ????
            instances.add(initInstance);
        }
        instances.setClassIndex(root.getInstance().classIndex());
        return instances;
    }

    private void processDistance(Instances instances, Instance instance, double distance, double[] distances) {
        int index = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < distances.length; i++) {
            if (max < distances[i]) {
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
