package structure.ballTree;

import structure.MathOperation;
import structure.Tree;
import structure.kdtree.KdTreeNode;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.*;

public class BallTree extends NearestNeighbourSearch implements Tree {
    private BallTreeNode root = null;
    private int classIndex = -1;
    private int k;
    private static final Random RANDOM = new Random();

    public BallTree(int k) {
        this.k = k;
    }

    @Override
    public void buildTree(Instances data) {

        Queue<BallTreeNode> nodeQueue = new LinkedList<>();
        classIndex = data.classIndex();
        Instance centroid1 = getCentroid(data);
        root = new BallTreeNode(centroid1);
        root.setRadius(getRadius(centroid1, data));
        root.setInstances(data);
        BallTreeNode node = root;
        nodeQueue.add(node);

        while (!nodeQueue.isEmpty()) {
            if (node == null) {
                System.err.println("Node is null");
                return;
            }
            Instances left = new Instances("left", getALlAttributes(data.firstInstance()), 0);
            Instances right = new Instances("right", getALlAttributes(data.firstInstance()), 0);

            node = nodeQueue.poll();
            if (!node.isInstances()) {
                System.err.println("Instances are not set");
                return;
            }
            data = node.getInstances();


            int randomIndex = getRandomIndex(data);
            Instance x0 = data.instance(randomIndex);           //3
            Instance x1 = getFarthestDistance(data, x0);        //4
            Instance x2 = getFarthestDistance(data, x1);        //5
            double[][] z = processProjection(x1, x2, data);     //6
            Arrays.sort(z, Comparator.comparingDouble(a -> a[0]));
            double m = z[z.length / 2][0];                      //7

            if (isAllSame(z)) {
                for (int i = 0; i < z.length; i++) {
                    if (i < z.length / 2) left.add(data.get((int) z[i][1]));
                    else right.add(data.get((int) z[i][1]));
                }
            } else {
                for (double[] pair : z) {
                    if (pair[0] < m) left.add(data.get((int) pair[1])); //8
                    else right.add(data.get((int) pair[1]));            //9
                }
            }
            if (left.size() > 0) {
                BallTreeNode leftNode = returnNode(left);
                node.setLeftSon(leftNode);
                if (left.size() > k) //else node is leaf
                    nodeQueue.add(leftNode);
            }
            if (right.size() > 0) {
                BallTreeNode rightNode = returnNode(right);
                node.setRightSon(rightNode);
                if (right.size() > k) //else node is leaf
                    nodeQueue.add(rightNode);
            }
            node.clearInstances();
        }
        //inOrderPrint();
        levelOrderPrint();
        System.out.println();
    }

    @Override
    public Instances findKNearestNeighbours(Instance instance, int k) {
        return null;
    }


    @Override
    public Instance nearestNeighbour(Instance target) throws Exception {
        return null;
    }

    @Override
    public Instances kNearestNeighbours(Instance target, int k) throws Exception {
        return null;
    }

    @Override
    public double[] getDistances() {
        return new double[0];
    }

    @Override
    public void update(Instance ins) throws Exception {

    }

    @Override
    public String getRevision() {
        return null;
    }

    private boolean isAllSame(double[][] z) {
        double val = z[0][0];
        for (int i = 1; i < z.length; i++) {
            if (z[i][0] != val)
                return false;
        }
        return true;
    }

    private void levelOrderPrint() {
        Queue<BallTreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            BallTreeNode tempNode = queue.poll();
            System.out.println("Centroid: " + tempNode.getCentroid());
            if (tempNode.isLeaf()) {
                System.out.println("-----DATA--------");
                Instances instances = tempNode.getInstances();
                for (Instance instance : instances) {
                    System.out.println(instance);
                }
                System.out.println("-------------");
            }
            if (tempNode.getLeftSon() != null) {
                queue.add(tempNode.getLeftSon());
            }

            if (tempNode.getRightSon() != null) {
                queue.add(tempNode.getRightSon());
            }
        }
    }


    public void inOrderPrint() {
        System.out.println("--------------------IN ORDER-------------------------");
        Stack<BallTreeNode> stack = new Stack<>();
        BallTreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                node = node.getLeftSon();
            } else {
                node = stack.pop();
                System.out.println(node.getCentroid());
                if (node.isInstances()) {
                    System.out.println("\t------------");
                    Instances instances = node.getInstances();
                    for (int i = 0; i < instances.size(); i++) {
                        System.out.print("\t" + instances.instance(i));
                    }
                    System.out.println("\n\t------------");
                }
                node = node.getRightSon();
            }
        }
        System.out.println("--------------------IN ORDER-------------------------");
    }

    private BallTreeNode returnNode(Instances data) {
        Instance centroid = getCentroid(data);
        double radius = getRadius(centroid, data);
        BallTreeNode node = new BallTreeNode(centroid);
        node.setRadius(radius);
        node.setInstances(data);
        return node;
    }

    private double getRadius(Instance centroid, Instances data) {
        double max = -Double.MIN_VALUE;
        for (Instance datum : data) {
            double distance = MathOperation.euclidDistance(centroid, datum);
            if (distance > max) {
                max = distance;
            }
        }
        return max;
    }

    private double[][] processProjection(Instance x1, Instance x2, Instances data) {
        double[] vector = new double[x1.numAttributes()];
        double[][] result = new double[data.size()][2];
        for (int i = 0; i < x1.numAttributes(); i++) {
            vector[i] = x1.value(i) - x2.value(i);
        }
        for (int i = 0; i < data.size(); i++) {
            double sum = 0;
            for (int j = 0; j < vector.length; j++) {
                sum += (vector[j] * data.get(i).value(j));
            }
            result[i][0] = sum;
            result[i][1] = i;
        }
        return result;
    }


    private Instance getCentroid(Instances data) {
        double[] values = new double[data.numAttributes()];
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < values.length; j++) {
                values[j] += data.get(i).value(j);
            }
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= data.size();
        }
        return new DenseInstance(1d, values);
    }

    private Instance getFarthestDistance(Instances data, Instance p_instance) {
        Instance instance = null;
        double max = -Double.MIN_VALUE;
        for (Instance inst : data) {
            if (inst == p_instance)
                continue;
            double distance = MathOperation.euclidDistance(inst, p_instance); //@TODO with or without class index
            if (max < distance) {
                instance = inst;
                max = distance;
            }
        }
        return instance;
    }

    private int getRandomIndex(Instances data) {
        int index = 0;
        index = RANDOM.nextInt(data.size());
        return index;
    }


    @Override
    public ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }


}
