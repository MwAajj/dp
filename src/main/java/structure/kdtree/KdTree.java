package structure.kdtree;

import structure.Tree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class KdTree implements Tree {
    private KdTreeNode root = null;
    private int NODES_SIZE = 2;
    //pick random attribute ???
    // urcis si pocet instancii v kazdom liste


    //mediany hladam vzdy len pre tie instancie ktore su mensie
    //ukladam si instanciu mediany nie median

    @Override
    public void buildTree(Instances data) {
        root = new KdTreeNode(data.firstInstance());
        Queue<KdTreeNode> nodeQueue = new LinkedList<>();

        KdTreeNode node;
        LinkedList<Instances> list = new LinkedList<>();
        list.add(data);
        Instance instance = getMedianInstance(new Instances(data), 0);

        root.setInstance(instance);
        root.setLevel(0);
        nodeQueue.add(root);

        //@TODO somehow improve memory management for list

        for (int i = 0; i < list.size(); i++) {
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
                Instance leftInstance = getMedianInstance(leftInstances, level);
                node.setLeftSon(new KdTreeNode(leftInstance));
                node.getLeftSon().setLevel(level);
                node.getLeftSon().setParent(node);
                nodeQueue.add(node.getLeftSon());
            }

            if (rightInstances.size() > 0) {
                level = getLevel(node);
                Instance rightInstance = getMedianInstance(rightInstances, level);
                node.setRightSon(new KdTreeNode(rightInstance));
                node.getRightSon().setLevel(level);
                node.getRightSon().setParent(node);
                nodeQueue.add(node.getRightSon());
            }
        }
        System.out.println("End");
    }

    private void inOrder() {

    }

    private int getLevel(KdTreeNode node) {
        int level = node.getLevel();
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

            if (compareInstances(instance, medianInstance) && !setMedium) {
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
            throw new RuntimeException("Incompatible size of instances");
        for (int i = 0; i < first.numAttributes(); i++) {
            if (first.value(i) != second.value(i))
                return false;
        }
        return true;
    }

    //ukladat si listy ktore mam prehladat
    @Override
    public double classifyInstance(Instance instance, int k) {
        return 0d;
    }

    private Instance getMedianInstance(Instances data, int level) {
        Instances instances = new Instances(data);
        instances.sort(level);
        return instances.get(instances.size() / 2);
    }

    private ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }
}
