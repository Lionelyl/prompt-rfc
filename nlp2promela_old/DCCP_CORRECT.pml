mtype = { DCCP_REQUEST, DCCP_RESPONSE, DCCP_DATA, DCCP_ACK, DCCP_DATAACK, DCCP_CLOSEREQ, DCCP_CLOSE, DCCP_RESET, DCCP_SYNC, DCCP_SYNCACK }
chan AtoN = [1] of { mtype }
chan NtoA = [0] of { mtype }
chan BtoN = [1] of { mtype }
chan NtoB = [0] of { mtype }

active proctype network() {
	do
	:: AtoN ? DCCP_REQUEST -> 
		if
		:: NtoB ! DCCP_REQUEST;
		fi unless timeout;

	:: AtoN ? DCCP_RESPONSE -> 
		if
		:: NtoB ! DCCP_RESPONSE;
		fi unless timeout;

	:: AtoN ? DCCP_DATA -> 
		if
		:: NtoB ! DCCP_DATA;
		fi unless timeout;

	:: AtoN ? DCCP_ACK -> 
		if
		:: NtoB ! DCCP_ACK;
		fi unless timeout;

	:: AtoN ? DCCP_DATAACK -> 
		if
		:: NtoB ! DCCP_DATAACK;
		fi unless timeout;

	:: AtoN ? DCCP_CLOSEREQ -> 
		if
		:: NtoB ! DCCP_CLOSEREQ;
		fi unless timeout;

	:: AtoN ? DCCP_CLOSE -> 
		if
		:: NtoB ! DCCP_CLOSE;
		fi unless timeout;

	:: AtoN ? DCCP_RESET -> 
		if
		:: NtoB ! DCCP_RESET;
		fi unless timeout;

	:: AtoN ? DCCP_SYNC -> 
		if
		:: NtoB ! DCCP_SYNC;
		fi unless timeout;

	:: AtoN ? DCCP_SYNCACK -> 
		if
		:: NtoB ! DCCP_SYNCACK;
		fi unless timeout;

	:: BtoN ? DCCP_REQUEST -> 
		if
		:: NtoA ! DCCP_REQUEST;
		fi unless timeout;

	:: BtoN ? DCCP_RESPONSE -> 
		if
		:: NtoA ! DCCP_RESPONSE;
		fi unless timeout;

	:: BtoN ? DCCP_DATA -> 
		if
		:: NtoA ! DCCP_DATA;
		fi unless timeout;

	:: BtoN ? DCCP_ACK -> 
		if
		:: NtoA ! DCCP_ACK;
		fi unless timeout;

	:: BtoN ? DCCP_DATAACK -> 
		if
		:: NtoA ! DCCP_DATAACK;
		fi unless timeout;

	:: BtoN ? DCCP_CLOSEREQ -> 
		if
		:: NtoA ! DCCP_CLOSEREQ;
		fi unless timeout;

	:: BtoN ? DCCP_CLOSE -> 
		if
		:: NtoA ! DCCP_CLOSE;
		fi unless timeout;

	:: BtoN ? DCCP_RESET -> 
		if
		:: NtoA ! DCCP_RESET;
		fi unless timeout;

	:: BtoN ? DCCP_SYNC -> 
		if
		:: NtoA ! DCCP_SYNC;
		fi unless timeout;

	:: BtoN ? DCCP_SYNCACK -> 
		if
		:: NtoA ! DCCP_SYNCACK;
		fi unless timeout;

	od
}
active proctype peerA(){
	goto CLOSED;
CHANGING:
	if
	:: skip;
	fi
CLOSED:
	if
	:: goto LISTEN;
	:: AtoN ! DCCP_REQUEST; goto REQUEST;
	fi
CLOSEREQ:
	if
	:: NtoA ? DCCP_CLOSE; AtoN ! DCCP_RESET; goto CLOSED;
	:: NtoA ? DCCP_CLOSEREQ; goto CLOSING;
	fi
CLOSING:
	if
	:: NtoA ? DCCP_RESET; goto TIMEWAIT;
	fi
LISTEN:
	if
	:: NtoA ? DCCP_REQUEST; AtoN ! DCCP_RESPONSE; goto RESPOND;
	:: goto CLOSED;
	fi
OPEN:
	if
	:: AtoN ! DCCP_DATA; goto OPEN;
	:: AtoN ! DCCP_DATAACK; goto OPEN;
	:: NtoA ? DCCP_ACK; goto OPEN;
	:: NtoA ? DCCP_DATA; goto OPEN;
	:: NtoA ? DCCP_DATAACK; goto OPEN;
	:: AtoN ! DCCP_CLOSEREQ; goto CLOSEREQ;
	:: NtoA ? DCCP_CLOSE; AtoN ! DCCP_RESET; goto CLOSED;
	:: AtoN ! DCCP_CLOSE; goto CLOSING;
	:: NtoA ? DCCP_CLOSEREQ; AtoN ! DCCP_CLOSE; goto CLOSING;
	fi
PARTOPEN:
	if
	:: AtoN ! DCCP_ACK; goto PARTOPEN;
	:: NtoA ? DCCP_DATA; AtoN ! DCCP_ACK; goto OPEN;
	:: NtoA ? DCCP_DATAACK; AtoN ! DCCP_ACK; goto OPEN;
	:: AtoN ! DCCP_DATAACK; goto PARTOPEN;
	:: goto CLOSED;
	:: NtoA ? DCCP_CLOSEREQ; AtoN ! DCCP_CLOSE; goto CLOSING;
	:: NtoA ? DCCP_CLOSE; AtoN ! DCCP_RESET; goto CLOSED;
	:: NtoA ? DCCP_REQUEST; goto OPEN;
	:: NtoA ? DCCP_DATA; goto OPEN;
	:: NtoA ? DCCP_ACK; goto OPEN;
	:: NtoA ? DCCP_DATAACK; goto OPEN;
	fi
REQUEST:
	if
	:: NtoA ? DCCP_RESPONSE; AtoN ! DCCP_ACK; goto PARTOPEN;
	:: NtoA ? DCCP_RESET; goto CLOSED;
	:: NtoA ? DCCP_SYNC; AtoN ! DCCP_RESET; goto CLOSED;
	fi
RESPOND:
	if
	:: NtoA ? DCCP_ACK; goto OPEN;
	:: NtoA ? DCCP_DATAACK; goto OPEN;
	:: AtoN ! DCCP_RESET; goto CLOSED;
	fi
STABLE:
	if
	:: skip;
	fi
TIMEWAIT:
	if
	:: goto CLOSED;
	fi
UNSTABLE:
	if
	:: skip;
	fi
}
active proctype peerB(){
	goto CLOSED;
CHANGING:
	if
	:: skip;
	fi
CLOSED:
	if
	:: goto LISTEN;
	:: BtoN ! DCCP_REQUEST; goto REQUEST;
	fi
CLOSEREQ:
	if
	:: NtoB ? DCCP_CLOSE; BtoN ! DCCP_RESET; goto CLOSED;
	:: NtoB ? DCCP_CLOSEREQ; goto CLOSING;
	fi
CLOSING:
	if
	:: NtoB ? DCCP_RESET; goto TIMEWAIT;
	fi
LISTEN:
	if
	:: NtoB ? DCCP_REQUEST; BtoN ! DCCP_RESPONSE; goto RESPOND;
	:: goto CLOSED;
	fi
OPEN:
	if
	:: BtoN ! DCCP_DATA; goto OPEN;
	:: BtoN ! DCCP_DATAACK; goto OPEN;
	:: NtoB ? DCCP_ACK; goto OPEN;
	:: NtoB ? DCCP_DATA; goto OPEN;
	:: NtoB ? DCCP_DATAACK; goto OPEN;
	:: BtoN ! DCCP_CLOSEREQ; goto CLOSEREQ;
	:: NtoB ? DCCP_CLOSE; BtoN ! DCCP_RESET; goto CLOSED;
	:: BtoN ! DCCP_CLOSE; goto CLOSING;
	:: NtoB ? DCCP_CLOSEREQ; BtoN ! DCCP_CLOSE; goto CLOSING;
	fi
PARTOPEN:
	if
	:: BtoN ! DCCP_ACK; goto PARTOPEN;
	:: NtoB ? DCCP_DATA; BtoN ! DCCP_ACK; goto OPEN;
	:: NtoB ? DCCP_DATAACK; BtoN ! DCCP_ACK; goto OPEN;
	:: BtoN ! DCCP_DATAACK; goto PARTOPEN;
	:: goto CLOSED;
	:: NtoB ? DCCP_CLOSEREQ; BtoN ! DCCP_CLOSE; goto CLOSING;
	:: NtoB ? DCCP_CLOSE; BtoN ! DCCP_RESET; goto CLOSED;
	:: NtoB ? DCCP_REQUEST; goto OPEN;
	:: NtoB ? DCCP_DATA; goto OPEN;
	:: NtoB ? DCCP_ACK; goto OPEN;
	:: NtoB ? DCCP_DATAACK; goto OPEN;
	fi
REQUEST:
	if
	:: NtoB ? DCCP_RESPONSE; BtoN ! DCCP_ACK; goto PARTOPEN;
	:: NtoB ? DCCP_RESET; goto CLOSED;
	:: NtoB ? DCCP_SYNC; BtoN ! DCCP_RESET; goto CLOSED;
	fi
RESPOND:
	if
	:: NtoB ? DCCP_ACK; goto OPEN;
	:: NtoB ? DCCP_DATAACK; goto OPEN;
	:: BtoN ! DCCP_RESET; goto CLOSED;
	fi
STABLE:
	if
	:: skip;
	fi
TIMEWAIT:
	if
	:: goto CLOSED;
	fi
UNSTABLE:
	if
	:: skip;
	fi
}
