library(igraph)
seed = 3

edges = c(1,2, 1,3, 1,4, 2,1, 2,3, 3,1, 3,2, 3,4, 4,1, 4,3, 4,5, 4,6, 
          5,4, 5,6, 5,7, 5,8, 6,4, 6,5, 6,7, 6,8, 7,5, 7,6, 7,8, 7,9, 
          8,5, 8,6, 8,7, 9,7)
g = make_graph(edges, directed=FALSE)
plot(g)

g = simplify(g)
set.seed(seed)
l = layout_with_fr(g)
plot(g, layout=l)

memb = rep(1, length(V(g)))
Q = modularity(g, membership=memb)
print(Q)

# create figure
# omi ('c(bottom, left, top, right)') and plt ('c(x1, x2, y1, y2)') 
# omi sizes the outer margins in inches
# plt gives the coordinates of the plot region as a fraction of the figure region
pdf('modularity-graphs.pdf')
par(mfrow=c(2,2), omi=c(0.5,0.3,0,0), plt=c(0.1,0.9,0,0.7))

# plot subplot(1, 1)
memb = c(1, 1, 1, 1, 1, 1, 1, 1, 1)
Q = modularity(g, membership=memb)
set.seed(seed)
l = layout_with_fr(g)
V(g)$color = memb
plot(g, layout=l, main = paste("Q:", round(Q, 3)), 
     vertex.size=20, edge.width=1.5, vertex.label=NA)
print(Q)

# plot subplot(1, 2)
memb = c(1, 1, 1, 2, 2, 2, 2, 2, 2)
Q = modularity(g, membership=memb)
set.seed(seed)
l = layout_with_fr(g)
V(g)$color = memb
plot(g, layout=l, main = paste("Q:", round(Q, 3)), 
     vertex.size=20, edge.width=1.5, vertex.label=NA)
print(Q)

# plot subplot(2, 1)
memb = c(1, 1, 1, 1, 2, 2, 2, 2, 2)
Q = modularity(g, membership=memb)
set.seed(seed)
l = layout_with_fr(g)
V(g)$color = memb
plot(g, layout=l, main = paste("Q:", round(Q, 3)), 
     vertex.size=20, edge.width=1.5, vertex.label=NA)
print(Q)

# plot subplot(2, 2)
memb = c(1, 1, 1, 1, 2, 2, 2, 2, 1)
Q = modularity(g, membership=memb)
set.seed(seed)
l = layout_with_fr(g)
V(g)$color = memb
plot(g, layout=l, main = paste("Q:", round(Q, 3)), 
     vertex.size=20, edge.width=1.5, vertex.label=NA)
print(Q)
dev.off()


