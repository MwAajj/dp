package structure.trees.ballTree;

import structure.EuclidDistance;
import structure.Structure;
import structure.trees.Son;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.*;
import java.util.Queue;

public class BallTree extends NearestNeighbourSearch implements Structure {
    private BallTreeNode root = null;
    private int classIndex = -1;
    private int numInst = -1;
    private final int k;
    PriorityQueue<DistInst> queue;
    private static final Random RANDOM = new Random(0);

    public BallTree(int k) {
        this.k = k;
    }

    public BallTree() {
        this.k = 2;
    }

    @Override
    public void buildStructure(Instances data) {
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
                if (node.isLeaf()) {
                    processLeaf(node, queue, target, k);
                    node = null; //leaf was checked
                    visited.push(Son.BOTH);
                } else {
                    left = node.getLeftSon() == null || visitedSon == Son.LEFT ? Double.MAX_VALUE
                            : EuclidDistance.euclidDistance(target, node.getLeftSon().getCentroid()) + node.getLeftSon().getRadius();
                    right = node.getRightSon() == null || visitedSon == Son.RIGHT ? Double.MAX_VALUE
                            : EuclidDistance.euclidDistance(target, node.getRightSon().getCentroid()) + node.getRightSon().getRadius();
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
                if (node.isLeaf() || isAllVisited(node, visitedSon)) {
                    node = null; //prevent from looping
                    continue;
                }
                d1 = EuclidDistance.euclidDistance(target, node.getCentroid()) - node.getRadius();
                d2 = queue.isEmpty() || queue.size() < k ? Double.MAX_VALUE
                        : EuclidDistance.euclidDistance(target, queue.peek().getInstance());
                if (d2 < d1) {
                    node = null; //there is no hope
                }

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
            d3 = EuclidDistance.euclidDistance(target, node.getInstances().get(i));
            if (queue.isEmpty()) d4 = Double.MAX_VALUE;
            else d4 = EuclidDistance.euclidDistance(target, queue.peek().getInstance());
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
                    double distance = EuclidDistance.euclidDistance(target, node.getInstances().get(i));
                    if (min > distance) {
                        min = distance;
                        instance = node.getInstances().get(i);
                    }
                }
                queue.add(new DistInst(instance, min));
                return instance;
            }
            left = node.getLeftSon() == null ? Double.MAX_VALUE :
                    EuclidDistance.euclidDistance(target, node.getLeftSon().getCentroid());
            right = node.getRightSon() == null ? Double.MAX_VALUE :
                    EuclidDistance.euclidDistance(target, node.getLeftSon().getCentroid());
            if (right == Double.MAX_VALUE && left == Double.MAX_VALUE)
                throw new RuntimeException("Unexpected Ball structure 1234");
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
        //Utils.normalize(distances);
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
            leftDistance = node.getLeftSon() == null ? Double.MAX_VALUE : EuclidDistance.euclidDistance(node.getLeftSon().getCentroid(), ins);
            rightDistance = node.getRightSon() == null ? Double.MAX_VALUE : EuclidDistance.euclidDistance(node.getRightSon().getCentroid(), ins);
            node = leftDistance < rightDistance ? node.getLeftSon() : node.getRightSon();
        }
    }

    private void splitInstances(Instances left, Instances right, Instances instances) {
        int randomIndex = RANDOM.nextInt(instances.size());
        Instance x0 = instances.instance(randomIndex);           //3
        Instance x1 = getFarthestDistance(instances, x0);        //4
        Instance x2 = getFarthestDistance(instances, x1);        //5
        double[][] z = processProjection(x1, x2, instances);     //6
        Arrays.sort(z, Comparator.comparingDouble(a -> a[0]));   //7a
        double m = getMedian(z);                                 //7b


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

    private double getMedian(double[][] z) {
        double m;
        int length = z.length;
        if (length % 2 == 1) {
            m = z[length / 2][0];
        } else {
            double sum = z[length / 2][0] + z[length / 2 - 1][0];
            m = sum / 2;
        }
        return m;
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
        for (Instance inst : data) {
            double distance = EuclidDistance.euclidDistance(inst, centroid);
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
        for (Instance instance : data) {
            for (int j = 0; j < values.length; j++) {
                //if (j == classIndex) continue;
                values[j] += instance.value(j);
            }
        }
        for (int i = 0; i < values.length; i++) {
            //if (i == classIndex) continue;
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
            double d2 = EuclidDistance.euclidDistance(inst, p_instance);
            double d1 = EuclidDistance.euclidDistance(p_instance, inst);
            double distance = Math.max(d2, d1); // note
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
