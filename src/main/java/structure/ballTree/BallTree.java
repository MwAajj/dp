package structure.ballTree;

import lombok.AllArgsConstructor;
import lombok.Getter;
import structure.MathOperation;
import structure.Tree;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.*;
import java.util.Queue;

public class BallTree extends NearestNeighbourSearch implements Tree {
    private BallTreeNode root = null;
    private int classIndex = -1;
    private int numInst = -1;
    private final int k;
    PriorityQueue<DistInst> queue;
    private static final Random RANDOM = new Random();

    public BallTree(int k) {
        this.k = k;
    }

    @Override
    public void buildTree(Instances data) {
        numInst = data.size();
        Queue<BallTreeNode> nodeQueue = new LinkedList<>();
        classIndex = data.classIndex();
        Instance centroid1 = getCentroid(data);
        root = new BallTreeNode(centroid1);
        root.setRadius(getRadius(centroid1, data));
        root.setInstances(data);
        BallTreeNode node = root;
        nodeQueue.add(node);

        while (!nodeQueue.isEmpty()) {
            node = nodeQueue.poll();
            checkNode(node);
            data = node.getInstances();

            Instances left = new Instances("left", getALlAttributes(data.firstInstance()), 0);
            Instances right = new Instances("right", getALlAttributes(data.firstInstance()), 0);
            splitInstances(left, right, data);
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
    }

    private void checkNode(BallTreeNode node) {
        if (node == null) {
            throw new RuntimeException("Node is null here");
        }

        if (!node.isInstances()) {
            throw new RuntimeException("Instances are not set");
        }
    }

    @Override
    public Instances findKNearestNeighbours(Instance target, int k) {
        checkData(k);
        BallTreeNode node = this.root;
        Son visitedSon = null;
        queue = new PriorityQueue<>(k);
        Stack<BallTreeNode> stack = new Stack<>();
        Stack<Son> visited = new Stack<>();
        double left, right, d1, d2;
        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                d1 = MathOperation.euclidDistance(classIndex, target, node.getCentroid()) - node.getRadius();
                d2 = queue.isEmpty() ? Double.MAX_VALUE
                        : MathOperation.euclidDistance(classIndex, target, queue.peek().getInstance());
                if (d1 >= d2 && queue.size() == k) {
                    return getInstancesQueue(target);
                }
                if (node.isLeaf()) {
                    processLeaf(node, queue, target, k);
                    node = null; //leaf was checked
                    visited.push(Son.BOTH);
                } else {
                    left = node.getLeftSon() == null || visitedSon == Son.LEFT ? Double.MAX_VALUE
                            : MathOperation.euclidDistance(classIndex, target, node.getLeftSon().getCentroid());
                    right = node.getRightSon() == null || visitedSon == Son.RIGHT ? Double.MAX_VALUE
                            : MathOperation.euclidDistance(classIndex, target, node.getRightSon().getCentroid());
                    if (left < right) {
                        node = node.getLeftSon();
                        if (visitedSon == Son.RIGHT) {
                            visited.push(Son.BOTH);
                        } else {
                            visited.push(Son.LEFT);
                        }
                    } else {
                        node = node.getRightSon();
                        if (visitedSon == Son.LEFT) {
                            visited.push(Son.BOTH);
                        } else {
                            visited.push(Son.RIGHT);
                        }
                    }
                }
                visitedSon = Son.NONE;
            } else {
                node = stack.pop();
                visitedSon = visited.pop();
                if (node.isLeaf() || isAllVisited(node, visitedSon))
                    node = null; //prevent from looping
            }
        }
        return getInstancesQueue(target);
    }

    private boolean isAllVisited(BallTreeNode node, Son son) {
        if (node.getLeftSon() != null && node.getRightSon() != null) {
            return son == Son.BOTH;
        }
        if (node.getLeftSon() != null && node.getRightSon() == null) {
            return son == Son.LEFT;
        }
        if (node.getLeftSon() == null && node.getRightSon() != null) {
            return son == Son.RIGHT;
        }
        return false;
    }

    private void checkData(int k) {
        if (numInst < k)
            throw new RuntimeException("K is bigger than data");
    }

    private Instances getInstancesQueue(Instance target) {
        Instances instances = new Instances("neighbours", getALlAttributes(target), k);
        for (DistInst distInst : queue) {
            instances.add(distInst.getInstance());
        }
        instances.setClassIndex(classIndex);
        return instances;
    }

    private void processLeaf(BallTreeNode node, PriorityQueue<DistInst> queue, Instance target, int k) {
        double d3, d4;
        if (node.getInstances().size() == 0)
            System.out.println("Unexpected processLeaf");
        for (int i = 0; i < node.getInstances().size(); i++) {
            d3 = MathOperation.euclidDistance(classIndex, target, node.getInstances().get(i));
            if (queue.isEmpty()) d4 = Double.MAX_VALUE;
            else d4 = MathOperation.euclidDistance(classIndex, target, queue.peek().getInstance());
            if (queue.size() < k)
                queue.add(new DistInst(node.getInstances().get(i), d3));
            else if (d3 < d4) {
                queue.add(new DistInst(node.getInstances().get(i), d3));
            }
            if (queue.size() > k)
                queue.poll();
        }
    }




    @Override
    public Instance nearestNeighbour(Instance target) {
        queue = new PriorityQueue<>(1);
        BallTreeNode node = this.root;
        double left, right;
        while (true) {
            if (node.isLeaf()) {
                double min = Double.MAX_VALUE;
                Instance instance = null;
                for (int i = 0; i < node.getInstances().size(); i++) {
                    double distance = MathOperation.euclidDistance(classIndex, target, node.getInstances().get(i));
                    if (min > distance) {
                        min = distance;
                        instance = node.getInstances().get(i);
                    }
                }
                queue.add(new DistInst(instance, min));
                return instance;
            }
            left = node.getLeftSon() == null ? Double.MAX_VALUE :
                    MathOperation.euclidDistance(classIndex, target, node.getLeftSon().getCentroid());
            right = node.getRightSon() == null ? Double.MAX_VALUE :
                    MathOperation.euclidDistance(classIndex, target, node.getLeftSon().getCentroid());
            if (right == Double.MAX_VALUE && left == Double.MAX_VALUE)
                throw new RuntimeException("Unexpected Ball tree 1234");
            node = left < right ? node.getLeftSon() : node.getRightSon();
        }
    }

    @Override
    public Instances kNearestNeighbours(Instance target, int k) {
        return findKNearestNeighbours(target, k);
    }

    @Override
    public double[] getDistances() {
        double[] distances = new double[queue.size()];
        int i = 0;
        for (DistInst distInst : queue) {
            distances[i] = distInst.getDistance();
            i++;
        }
        return distances;
    }

    @Override
    public void update(Instance ins) {
        if (ins.classIndex() != classIndex)
            throw new RuntimeException("Incorrect class index in instance: " + ins);
        numInst++;
        if (root == null) {
            root = new BallTreeNode(ins);
            root.setCentroid(ins);
            root.setRadius(Double.MAX_VALUE);
            return;
        }
        BallTreeNode node = root;
        double leftDistance, rightDistance;
        while (true) {
            if (node.isLeaf()) {
                if (!node.isInstances())
                    throw new RuntimeException("Leaf without instance");
                if (node.getInstances().size() < k) {
                    node.addInstance(ins);
                    return;
                }
                Instances instances = node.getInstances();
                instances.add(ins);
                node.clearInstances();

                Instances left = new Instances("left", getALlAttributes(instances.firstInstance()), 0);
                Instances right = new Instances("right", getALlAttributes(instances.firstInstance()), 0);
                splitInstances(left, right, instances);
                if (left.size() > 0) {
                    node.setLeftSon(returnNode(left));
                }
                if (right.size() > 0) {
                    node.setRightSon(returnNode(right));
                }
                return;
            }
            leftDistance = node.getLeftSon() == null ? Double.MAX_VALUE : MathOperation.euclidDistance(classIndex, node.getLeftSon().getCentroid(), ins);
            rightDistance = node.getRightSon() == null ? Double.MAX_VALUE : MathOperation.euclidDistance(classIndex, node.getRightSon().getCentroid(), ins);
            node = leftDistance < rightDistance ? node.getLeftSon() : node.getRightSon();
        }
    }

    private void splitInstances(Instances left, Instances right, Instances instances) {
        int randomIndex = RANDOM.nextInt(instances.size());
        Instance x0 = instances.instance(randomIndex);           //3
        Instance x1 = getFarthestDistance(instances, x0);        //4
        Instance x2 = getFarthestDistance(instances, x1);        //5
        double[][] z = processProjection(x1, x2, instances);     //6
        Arrays.sort(z, Comparator.comparingDouble(a -> a[0]));
        double m = z[z.length / 2][0];                      //7

        if (isAllSame(z)) { //special check preventing to cycle algorithm
            for (int i = 0; i < z.length; i++) {
                if (i < z.length / 2) left.add(instances.get((int) z[i][1]));
                else right.add(instances.get((int) z[i][1]));
            }
        } else {
            for (double[] pair : z) {
                if (pair[0] < m) left.add(instances.get((int) pair[1])); //8
                else right.add(instances.get((int) pair[1]));            //9
            }
        }
    }

    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 2 $");
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
            double distance = MathOperation.euclidDistance(classIndex, centroid, datum);
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
        for (Instance datum : data) {
            for (int j = 0; j < values.length; j++) {
                values[j] += datum.value(j);
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
            double distance = MathOperation.euclidDistance(classIndex, inst, p_instance);
            if (max < distance) {
                instance = inst;
                max = distance;
            }
        }
        return instance;
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
