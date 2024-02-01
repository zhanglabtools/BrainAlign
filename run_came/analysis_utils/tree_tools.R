#' Check whether object is a data tree
#'
#' @param x Object to test for tree structure
#' 
#' @return (logical scalar) TRUE or FALSE
isTree <- function(x){
  any(class(x) == "Node") & any(class(x) == "R6")
}


parse_abi_hierarchy <-
  function(abi){
    abi_list <- fromJSON(file = abi)$msg[[1]]
    tree <- recursive_build_tree(abi_list)
  }

recursive_build_tree <- function(tree_list){
  children_list <- tree_list$children
  node_list <- tree_list[names(tree_list) != "children"]
  
  node <- do.call(Node$new, node_list)
  
  if(is.null(children_list) || length(children_list) == 0){
    return(node)
  } else {
    lapply(children_list, function(cl){
      child_node <- recursive_build_tree(cl)
      node$AddChildNode(child_node)
      
      NULL
    })
    
    return(node)
  }
}


#' Attach cut points to a tree node
#' 
#' @description 
#' This function attaches cut points to a data tree node. The cut
#' points are used by the pruning functions pruneAtNode and 
#' pruneBelowNode to prune the tree based on those nodes.
#' 
#' @param node (Node) A data tree node 
#' @param cutPoints (character vector) The names of tree nodes to use 
#' as cut points.
#'                  
#' @return NULL                
attachCutPoints <- function(node, cutPoints){
  node[["cutPoints"]] <- cutPoints
}


#' Prune data tree at node
#' 
#' @description 
#' Prune the data tree at the specified node name.
#' 
#' @details 
#' This pruning function evaluates whether the name of the node is
#' in the cut points (stored in the node field `cutPoints`). When 
#' passed to the `pruneFun` argument in the `Prune` function from 
#' `data.tree` it will prune the data tree at the nodes specified 
#' in the `cutPoints`` field.
#'
#' @param node (Node) A data tree node.
#' 
#' @return (logical scalar) TRUE or FALSE 
pruneAtNode <- function(node){
  
  if(is.null(node[["cutPoints"]])){
    stop(paste("Node", node[["name"]], "has no field 'cutPoints'.",
               "Cut points can be attached using the function",
               "attachCutPoints()."))
  }
  
  return(!(node$name %in% node$cutPoints))
}


#' Prune data tree below node
#' 
#' @description 
#' Prune the data tree below the specified node name.
#' 
#' @details 
#' This pruning function evaluates whether the node is a child of 
#' the cut points (stored in the node field "cutPoints"). When 
#' passed to the `pruneFun` argument in the `Prune` function from 
#' `data.tree` it will prune the data tree below the nodes specified
#' in the "cutPoints" field.
#'
#' @param node (Node) A data tree node.
#' 
#' @return (logical scalar) TRUE or FALSE 
pruneBelowNode <- function(node){
  
  if(is.null(node[["cutPoints"]])){
    stop(paste("Node", node[["name"]], "has no field 'cutPoints'.",
               "Cut points can be attached using the function",
               "attachCutPoints()."))
  }
  
  return(!any(node$cutPoints %in% node$path[-length(node$path)]))
  }


#' Prune a tree based on node names
#' 
#' @description 
#' 
#' @param anatTree The data tree to prune.
#' @param nodes (character vector) Node names to use to prune the tree.
#' @param method (character scalar) One of 'AtNode' or 'BelowNode' 
#' indicating whether to prune the tree at or below the nodes specified.              
#' 
#' @return (character scalar) Message indicating how the tree was 
#' pruned.
pruneAnatTree <- function(anatTree, nodes, method = "AtNode"){
  
  require(data.tree)
  
  #Select where to prune
  if(method == "AtNode"){
    pruningFunction <- pruneAtNode
  } else if (method == "BelowNode"){
    pruningFunction <- pruneBelowNode
  } else {
    stop(paste("Argument `method` must be one of 'AtNode' or 'BelowNode'.",
               "Got", method))
  }
  
  #Attach cut points to tree
  anatTree$Do(attachCutPoints, cutPoints = nodes)
  
  #Prune cut points from tree 
  Prune(anatTree, pruningFunction)
  
  outMessage <- paste("Tree pruned",
                      ifelse(method == "AtNode", "at nodes", "below nodes"),
                      paste(nodes, collapse = ", "))
  
  return(outMessage) 
}


#' Create an atlas from a tree
#' 
#' @description 
#' Create an atlas in MINC format from the leaf nodes of a tree.
#'
#' @param anatTree A data tree.
#' @param labelVolume (mincSingleDim) 3-dimensional array containing
#' the all labels in the `label` field at the root level of `anatTree`.
#'
#' @return (mincSingleDim) 1-dimensional array containing the atlas 
#' labels.
hanatToAtlas <- function(anatTree, labelVolume){
  
  require(RMINC)
  require(data.tree)
  
  anatTree$Do(function(node){
    node$label.new <- min(node$label)
  }, traversal = "post-order")
  
  out <- hanatToVolume(anatTree, labelVolume, "label.new")
  out <- as.numeric(out)
  class(out) <- class(labelVolume)
  attributes(out) <- attributes(labelVolume)
  attr(out, "dim") <- NULL
  
  return(out)
}


#' Create atlas definitions.
#'
#' @description 
#' Generate atlas definitions based on the leaf nodes of a
#' neuroanatomical tree.
#'
#' @param anatTree A data tree.
#'
#' @return (data.frame) A data frame containing the definitions of the 
#' atlas labels associated with the leaf nodes of `anatTree`.
hanatToAtlasDefs <- function(anatTree){
  
  require(data.tree)
  
  if(is.null(anatTree[["label.new"]])){
    anatTree$Do(function(node){
      node$label.new <- min(node$label)
    }, traversal = "post-order")
  }
  
  labelsNew <- anatTree$Get("label.new", filterFun = isLeaf)
  
  return(data.frame(Structure = names(labelsNew),
                    Label = labelsNew,
                    row.names = NULL,
                    stringsAsFactors = FALSE))
}